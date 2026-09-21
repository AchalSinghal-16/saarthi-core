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
#  Terminal alert formatting
# ──────────────────────────────────────────────────────────────

# Cooldown (seconds) per object class to avoid spamming the same alert
ALERT_COOLDOWN_S = 2.0


def print_alert(det, alert_tracker: dict):
    """
    Print a structured spatial warning to the terminal, respecting
    a per-class cooldown to avoid flooding.

    Format:
        [ALERT] Chair detected 1.3m ahead in Center
    """
    now = time.time()
    key = f"{det.label}_{det.direction}"

    # Skip if we alerted for this same class+direction recently
    if key in alert_tracker and (now - alert_tracker[key]) < ALERT_COOLDOWN_S:
        return

    alert_tracker[key] = now

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

# Run face identification every N seconds (it's heavier than YOLO)
FACE_IDENTIFY_INTERVAL_S = 2.0


def draw_face_result(frame, face_result: dict, alert_tracker: dict):
    """Draw the face recognition result on the frame and print alerts."""
    if not face_result["detected"]:
        return

    name = face_result["name"]
    h, w = frame.shape[:2]

    if name == "Unknown":
        color = (0, 165, 255)  # Orange
        label = "Face: Unknown"
    else:
        color = (255, 200, 0)  # Cyan-ish
        dist_str = f" ({face_result['distance']:.3f})" if face_result["distance"] >= 0 else ""
        label = f"Face: {name}{dist_str}"

    # Draw label at bottom-left of the frame
    cv2.putText(
        frame, label, (10, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
    )

    # Terminal alert (with cooldown)
    now = time.time()
    alert_key = f"face_{name}"
    if alert_key not in alert_tracker or (now - alert_tracker[alert_key]) > 3.0:
        alert_tracker[alert_key] = now
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

    alert_tracker: dict = {}  # Cooldown tracker for terminal alerts
    prev_time = time.time()

    # ── Face recognition runs on a BACKGROUND THREAD ──
    # This prevents it from blocking the main video loop
    face_lock = threading.Lock()
    face_result_shared = {"name": "No Face", "distance": -1, "id": None, "detected": False}
    face_running = True

    def face_recognition_worker():
        """Background thread: periodically runs face identification."""
        nonlocal face_result_shared
        while face_running:
            frame = reader.read()
            if frame is not None:
                try:
                    result = face_memory.identify_face(frame)
                    with face_lock:
                        face_result_shared = result
                except Exception:
                    pass  # Silently skip on any error
            time.sleep(FACE_IDENTIFY_INTERVAL_S)

    enrolled_count = face_memory.count()
    print(f"\n  [Pipeline] {enrolled_count} face(s) enrolled in database.")
    print("  [Pipeline] Running. Press 'q' in the window to quit.\n")
    print("  ─── Terminal Alerts ─────────────────────────────")

    try:
        reader.start()

        # Start face recognition on a background thread
        face_thread = threading.Thread(target=face_recognition_worker, daemon=True)
        face_thread.start()

        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Obstacle Detection (every frame) ──
            detections = detector.detect(frame)
            hazards = detector.get_hazards(detections)

            # ── Terminal alerts for hazards ──
            for det in hazards:
                print_alert(det, alert_tracker)

            # ── Read latest face result (thread-safe) ──
            with face_lock:
                current_face_result = face_result_shared.copy()

            # ── FPS calculation ──
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # ── Visual overlay ──
            draw_sector_guides(frame)
            draw_detections(frame, detections)
            draw_face_result(frame, current_face_result, alert_tracker)
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
        face_running = False
        reader.stop()
        cv2.destroyAllWindows()
        print("  [Pipeline] Clean shutdown complete.\n")


if __name__ == "__main__":
    main()
