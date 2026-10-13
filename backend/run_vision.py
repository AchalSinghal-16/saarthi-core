"""
Saarthi — Vision Pipeline Runner
==================================
Integrates the threaded video stream reader with the obstacle detector,
spatial distance engine, and facial recognition system. Provides
real-time visual overlays and structured terminal alerts.

Usage:
    python backend/run_vision.py --source "http://<PHONE_IP>:8080/video"
    python backend/run_vision.py --source 0
"""

import sys
import time
import argparse
import threading
from datetime import datetime

import cv2

from vision.stream_reader import ThreadedStreamReader
from vision.obstacle_detector import ObstacleDetector, HAZARD_ZONE_M
from vision.face_memory import FaceMemory


# ──────────────────────────────────────────────────────────────
#  Alert Fatigue & Spatial Warning Manager
# ──────────────────────────────────────────────────────────────

class AudioCooldownMatrix:
    """
    Suppresses repetitive hazard warnings to prevent alert fatigue.

    Rules:
      1. Default cooldown of 10.0 seconds per (label, sector).
      2. Delta-distance override: If the obstacle is >= 0.5m closer
         than the last recorded distance, bypass the 10-second cooldown
         and alert immediately.
    """
    COOLDOWN_SECONDS = 10.0
    DELTA_DISTANCE_M = 0.5

    def __init__(self, cooldown_seconds: float = 10.0, delta_distance_m: float = 0.5):
        self.cooldown_seconds = cooldown_seconds
        self.delta_distance_m = delta_distance_m
        # Key: (label.lower(), sector) -> {"timestamp": float, "distance": float}
        self.last_alert_times = {}

    def should_alert(self, det) -> bool:
        """
        Evaluate if an alert should trigger.
        Returns True if:
          - Never alerted for this object & sector, OR
          - >= 10 seconds elapsed since last alert, OR
          - Obstacle moved closer by >= 0.5 meters.
        """
        now = time.time()
        key = (det.label.lower(), det.direction)

        if key not in self.last_alert_times:
            self.last_alert_times[key] = {
                "timestamp": now,
                "distance": det.distance_m,
            }
            return True

        record = self.last_alert_times[key]
        time_elapsed = now - record["timestamp"]
        prev_distance = record["distance"]

        # Delta-distance override: object moved closer by >= 0.5m
        is_closer_by_delta = (
            prev_distance > 0
            and det.distance_m > 0
            and (prev_distance - det.distance_m >= self.delta_distance_m)
        )

        if is_closer_by_delta or (time_elapsed >= self.cooldown_seconds):
            self.last_alert_times[key] = {
                "timestamp": now,
                "distance": det.distance_m,
            }
            return True

        return False

    def emit_alert(self, det):
        """Check cooldown rules and print spatial warning if permitted."""
        if self.should_alert(det):
            dist_str = f"{det.distance_m:.1f}m" if det.distance_m > 0 else "unknown distance"
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"  [{timestamp}] [ALERT] {det.label.capitalize()} detected "
                f"{dist_str} ahead in {det.direction}"
            )



# ──────────────────────────────────────────────────────────────
#  Visual overlay helpers
# ──────────────────────────────────────────────────────────────

def draw_sector_guides(frame):
    """Draw faint vertical lines at the 35% and 65% sector boundaries."""
    h, w = frame.shape[:2]
    left_x = int(w * 0.35)
    right_x = int(w * 0.65)
    color = (100, 100, 100)
    cv2.line(frame, (left_x, 0), (left_x, h), color, 1)
    cv2.line(frame, (right_x, 0), (right_x, h), color, 1)

    # Sector labels at the top
    cv2.putText(frame, "LEFT", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(frame, "CENTER", (left_x + 10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(frame, "RIGHT", (right_x + 10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_detections(frame, detections):
    """Overlay bounding boxes, labels, distance, and direction on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # Red for hazards (< 2.5m), green for safe
        if det.is_hazard:
            color = (0, 0, 255)       # Red
            thickness = 3
        else:
            color = (0, 255, 0)       # Green
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label:  "person 1.3m [Center]"
        dist_str = f"{det.distance_m:.1f}m" if det.distance_m > 0 else "?m"
        label_text = f"{det.label} {dist_str} [{det.direction}]"

        # Background rectangle for readability
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, label_text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )


def draw_hud(frame, total, hazard_count, fps):
    """Draw the heads-up display bar at the top of the frame."""
    hud_color = (0, 0, 255) if hazard_count > 0 else (0, 200, 0)

    # Semi-transparent HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    hud_text = (
        f"SAARTHI  |  Objects: {total}  |  "
        f"HAZARDS: {hazard_count}  |  FPS: {fps:.0f}"
    )
    cv2.putText(
        frame, hud_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2,
    )


# ──────────────────────────────────────────────────────────────
#  Face recognition overlay
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
#  Face recognition overlay
# ──────────────────────────────────────────────────────────────

FACE_IDENTIFY_INTERVAL_S = 1.0


def draw_face_boxes(frame, detected_faces: list, face_alert_tracker: dict):
    """
    Render bounding boxes and names directly on faces using cached MTCNN coordinates.

    Args:
        frame: OpenCV BGR frame.
        detected_faces: List of (name, (x1, y1, x2, y2)) tuples.
        face_alert_tracker: Dict tracking timestamps for terminal face alerts.
    """
    for name, bbox in detected_faces:
        x1, y1, x2, y2 = bbox

        if name == "Unknown":
            box_color = (0, 165, 255)   # Orange for unknown face
            text_color = (255, 255, 255)
            display_name = "Unknown"
        else:
            box_color = (0, 255, 0)     # Green for recognized face
            text_color = (0, 0, 0)
            display_name = name

        # Bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Name label badge above the bounding box
        label_text = f"Face: {display_name}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw filled label background
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), box_color, -1)

        # Text over the badge
        cv2.putText(
            frame, label_text, (x1 + 3, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
        )

        # Terminal face alerts (cooldown 5.0 seconds per person)
        now = time.time()
        alert_key = f"face_{name}"
        if alert_key not in face_alert_tracker or (now - face_alert_tracker[alert_key]) > 5.0:
            face_alert_tracker[alert_key] = now
            timestamp = datetime.now().strftime("%H:%M:%S")
            if name != "Unknown":
                print(f"  [{timestamp}] [FACE] Recognized: {name}")
            else:
                print(f"  [{timestamp}] [FACE] Unknown person detected")


# ──────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Saarthi Vision Pipeline — Live Hazard Perception"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Video source: a network stream URL "
            "(e.g. http://<PHONE_IP>:8080/video) "
            "or an integer camera index (default: 0)."
        ),
    )
    args = parser.parse_args()

    # Parse source
    source = int(args.source) if args.source.isdigit() else args.source

    # ── Startup ──
    print("=" * 55)
    print("   Saarthi — Live Hazard & Face Recognition Pipeline")
    print("=" * 55)
    print(f"  Source       : {source}")
    print(f"  Hazard zone  : < {HAZARD_ZONE_M} m")
    print()

    # Initialize components
    reader = ThreadedStreamReader(source=source)
    detector = ObstacleDetector()
    face_memory = FaceMemory()
    alert_matrix = AudioCooldownMatrix(cooldown_seconds=10.0, delta_distance_m=0.5)

    face_alert_tracker = {}
    prev_time = time.time()
    pipeline_running = True

    # ── Shared state (written by background threads, read by main) ──
    detection_lock = threading.Lock()
    detections_shared = []
    hazards_shared = []

    face_lock = threading.Lock()
    cached_faces = []  # List of (name, (x1, y1, x2, y2)) tuples

    # ────────────────────────────────────────────────────
    #  Background thread 1: YOLO obstacle detection
    # ────────────────────────────────────────────────────
    def yolo_worker():
        while pipeline_running:
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                small = cv2.resize(frame, (640, 480))
                dets = detector.detect(small)

                # Scale bboxes back to original size
                h_orig, w_orig = frame.shape[:2]
                sx, sy = w_orig / 640.0, h_orig / 480.0
                for d in dets:
                    x1, y1, x2, y2 = d.bbox
                    d.bbox = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

                haz = detector.get_hazards(dets)

                with detection_lock:
                    detections_shared.clear()
                    detections_shared.extend(dets)
                    hazards_shared.clear()
                    hazards_shared.extend(haz)

                # Process hazard alerts through AudioCooldownMatrix
                for det in haz:
                    alert_matrix.emit_alert(det)

            except Exception:
                pass
            time.sleep(0.03)

    # ────────────────────────────────────────────────────
    #  Background thread 2: Dedicated daemon face worker
    # ────────────────────────────────────────────────────
    def face_worker_thread():
        nonlocal cached_faces
        while pipeline_running:
            frame = reader.read()
            if frame is not None:
                try:
                    results = face_memory.identify_face(frame)
                    with face_lock:
                        cached_faces = list(results)
                except Exception:
                    pass
            time.sleep(FACE_IDENTIFY_INTERVAL_S)

    enrolled_count = face_memory.count()
    print(f"\n  [Pipeline] {enrolled_count} face(s) enrolled in database.")
    print("  [Pipeline] Running. Press 'q' in the window to quit.\n")
    print("  ─── Terminal Alerts ─────────────────────────────")

    try:
        reader.start()

        # Launch background daemon workers
        threading.Thread(target=yolo_worker, daemon=True).start()
        threading.Thread(target=face_worker_thread, daemon=True).start()

        # ── Main loop: ONLY display & overlays (zero inference latency) ──
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Read cached YOLO results (non-blocking)
            with detection_lock:
                detections = list(detections_shared)
                hazards = list(hazards_shared)

            # Read cached face recognition results (non-blocking)
            with face_lock:
                current_faces = list(cached_faces)

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Draw overlays
            draw_sector_guides(frame)
            draw_detections(frame, detections)
            draw_face_boxes(frame, current_faces, face_alert_tracker)
            draw_hud(frame, len(detections), len(hazards), fps)

            cv2.imshow("Saarthi — Hazard Perception", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except ConnectionError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  [Pipeline] Interrupted by user.")
    finally:
        pipeline_running = False
        reader.stop()
        cv2.destroyAllWindows()
        print("  [Pipeline] Clean shutdown complete.\n")


if __name__ == "__main__":
    main()
