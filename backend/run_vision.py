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
from collections import deque

import cv2
import keyboard
import speech_recognition as sr

from vision.stream_reader import ThreadedStreamReader
from vision.obstacle_detector import ObstacleDetector, HAZARD_ZONE_M
from vision.face_memory import FaceMemory
from vision.ocr_reader import OCRReader
from vision.currency_detector import CurrencyDetector
from audio.synthesizer import AudioSynthesizer
from navigation.locator import LocationManager


# ──────────────────────────────────────────────────────────────
#  Alert Fatigue & Spatial Warning Manager
# ──────────────────────────────────────────────────────────────

class AudioCooldownMatrix:
    """
    Suppresses repetitive hazard warnings to prevent alert fatigue.

    Rules:
      1. Default cooldown of 5.0 seconds per (label, sector).
      2. Delta-distance override: If the obstacle is >= 0.5m closer
         than the last recorded distance, bypass the 5-second cooldown
         and alert immediately.
    """
    COOLDOWN_SECONDS = 5.0
    DELTA_DISTANCE_M = 0.5

    def __init__(self, cooldown_seconds: float = 5.0, delta_distance_m: float = 0.5):
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

    def emit_alert(self, det, synthesizer=None):
        """Check cooldown rules, print spatial warning, and speak aloud if permitted."""
        if self.should_alert(det):
            dist_str = f"{det.distance_m:.1f}m" if det.distance_m > 0 else "unknown distance"
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"  [{timestamp}] [ALERT] {det.label.capitalize()} detected "
                f"{dist_str} ahead in {det.direction}"
            )
            if synthesizer is not None:
                synthesizer.speak(f"{det.label}, {det.direction}")



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


def draw_face_boxes(frame, detected_faces: list, face_alert_tracker: dict, synthesizer=None):
    """
    Render bounding boxes and names directly on faces using cached MTCNN coordinates,
    and announce recognized enrolled individuals aloud.

    Args:
        frame: OpenCV BGR frame.
        detected_faces: List of (name, (x1, y1, x2, y2)) tuples.
        face_alert_tracker: Dict tracking timestamps for face alerts.
        synthesizer: Optional AudioSynthesizer instance for spoken announcements.
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

        # Face alerts (cooldown 10.0 seconds per person)
        now = time.time()
        alert_key = f"face_{name.lower()}"
        if alert_key not in face_alert_tracker or (now - face_alert_tracker[alert_key]) > 10.0:
            face_alert_tracker[alert_key] = now
            timestamp = datetime.now().strftime("%H:%M:%S")
            if name != "Unknown":
                print(f"  [{timestamp}] [FACE] Recognized: {name}")
                if synthesizer is not None:
                    synthesizer.speak(f"{name} detected")
            else:
                print(f"  [{timestamp}] [FACE] Unknown person detected")


# ──────────────────────────────────────────────────────────────
#  Assistive On-Demand Overlays: OCR & Currency
# ──────────────────────────────────────────────────────────────

def wrap_text_to_lines(text: str, max_pixel_width: int, font_scale: float = 0.55, thickness: int = 1) -> list:
    """
    Word-wrap multi-paragraph text cleanly to ensure no line exceeds
    max_pixel_width when rendered using cv2.putText.
    """
    wrapped = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        words = paragraph.split(" ")
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if tw <= max_pixel_width:
                current_line.append(word)
            else:
                if current_line:
                    wrapped.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            wrapped.append(" ".join(current_line))
    return wrapped


def draw_currency_banner(frame, banner_text: str):
    """Flashes a prominent visual HUD banner across the frame for detected currency."""
    h, w = frame.shape[:2]
    # Semi-transparent dark banner bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 85), (w, h - 25), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Gold accent border
    cv2.rectangle(frame, (0, h - 85), (w, h - 25), (0, 215, 255), 2)
    cv2.putText(
        frame, banner_text, (20, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2
    )


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
    alert_matrix = AudioCooldownMatrix(cooldown_seconds=5.0, delta_distance_m=0.5)

    print("\n  [Init] Loading assistive VLM, Audio, and Currency modules...")
    ocr_reader = OCRReader()
    currency_detector = CurrencyDetector(ocr_reader=ocr_reader)
    synthesizer = AudioSynthesizer()
    location_manager = LocationManager()
    if isinstance(source, str) and "://" in source:
        location_manager.set_sensor_source(source)

    last_key_time = 0.0

    face_alert_tracker = {}
    prev_time = time.time()
    pipeline_running = True
    active_mode_active = False  # Controls pausing passive YOLO scanning during active queries

    # ── VLM Temporal Memory (5-second rolling window FIFO ring buffer) ──
    recent_frames = deque(maxlen=5)
    last_save_time = time.time()

    # ── Shared state (written by background threads, read by main) ──
    detection_lock = threading.Lock()
    detections_shared = []
    hazards_shared = []

    face_lock = threading.Lock()
    cached_faces = []  # List of (name, (x1, y1, x2, y2)) tuples

    # ── On-demand assistive feature state ──
    ocr_lock = threading.Lock()
    ocr_busy = False

    currency_lock = threading.Lock()
    cached_currency_banner = ""
    currency_expiry = 0.0
    currency_busy = False

    voice_lock = threading.Lock()
    voice_busy = False

    location_lock = threading.Lock()
    location_busy = False

    # State variables for on-screen HUD readout display
    display_lock = threading.Lock()
    display_text = ""
    display_timer = 0.0

    # Immediate action banner state
    status_lock = threading.Lock()
    status_text = ""
    status_expiry = 0.0

    def trigger_ocr(frame_to_scan):
        nonlocal ocr_busy, status_text, status_expiry, display_text, display_timer
        with ocr_lock:
            if ocr_busy:
                print("  [OCR] Already scanning with Gemini VLM, please wait...", flush=True)
                return
            ocr_busy = True

        # Immediately display scanning status on HUD without blocking video loop
        with display_lock:
            display_text = "Reading document with Gemini VLM... Please hold steady"
            display_timer = time.time() + 15.0
        with status_lock:
            status_text = "OCR: Reading text with Gemini VLM..."
            status_expiry = time.time() + 8.0

        def _ocr_background_task(frame):
            nonlocal ocr_busy, status_text, status_expiry, display_text, display_timer
            try:
                print("\n" + "=" * 55, flush=True)
                print("  [ACTION] Processing VLM Readout in Background Thread (Gemini 1.5 Flash)...", flush=True)
                print("=" * 55, flush=True)
                text_out = ocr_reader.read_text(frame)
                if text_out:
                    print(f"\n  [VLM READOUT]:\n{text_out}\n", flush=True)
                    with display_lock:
                        display_text = text_out
                        display_timer = time.time() + 12.0  # 12s for comfortable reading
                    with status_lock:
                        status_text = f"READ: {text_out.replace(chr(10), ' ')[:45]}..."
                        status_expiry = time.time() + 8.0
                    # Auto-play text immediately through synthesizer
                    synthesizer.speak(text_out, interrupt=True)
                else:
                    print("\n  [VLM READOUT]: No legible text detected in frame.\n", flush=True)
                    with display_lock:
                        display_text = "No legible text detected."
                        display_timer = time.time() + 5.0
                    with status_lock:
                        status_text = "OCR: No text detected (hold closer to camera)"
                        status_expiry = time.time() + 4.0
                    synthesizer.speak("No legible text detected.", interrupt=True)
            except Exception as e:
                print(f"  [OCR ERROR]: {e}", flush=True)
                synthesizer.speak("Error reading document.", interrupt=True)
            finally:
                with ocr_lock:
                    ocr_busy = False

        # Execute heavy Gemini API network call on background thread so video FPS never drops
        threading.Thread(target=_ocr_background_task, args=(frame_to_scan.copy(),), daemon=True).start()

    def trigger_currency(frame_to_scan):
        nonlocal currency_busy, cached_currency_banner, currency_expiry, status_text, status_expiry, display_text, display_timer
        with currency_lock:
            if currency_busy:
                print("  [CURRENCY] Already scanning, please wait...", flush=True)
                return
            currency_busy = True
        # Immediately display scanning status on HUD so user sees it right away
        with display_lock:
            display_text = "Currency: Scanning note... Please hold steady"
            display_timer = time.time() + 10.0
        with status_lock:
            status_text = "CURRENCY: Scanning note... Please hold steady"
            status_expiry = time.time() + 8.0
        synthesizer.speak("Scanning note.", interrupt=True)
        try:
            print("\n" + "=" * 55, flush=True)
            print("  [ACTION] Checking Indian Rupee Banknote...", flush=True)
            print("=" * 55, flush=True)
            denom, conf = currency_detector.detect_currency(frame_to_scan)
            if conf > 0:
                denomination = f"{denom} ({int(conf * 100)}%)"
            else:
                denomination = denom
            print(f"\n  [CURRENCY]: Currency Detected: {denomination}\n", flush=True)
            with display_lock:
                display_text = f"Currency Detected: {denomination}"
                display_timer = time.time() + 8.0  # Display for 8.0s
            with status_lock:
                status_text = f"CURRENCY: {denomination}"
                status_expiry = time.time() + 8.0
            spoken_denom = f"Currency detected: {denom}" if conf > 0 else denomination
            synthesizer.speak(spoken_denom, interrupt=True)
        except Exception as e:
            print(f"  [CURRENCY ERROR]: {e}", flush=True)
            synthesizer.speak("Could not identify currency note.", interrupt=True)
        finally:
            with currency_lock:
                currency_busy = False

    def trigger_active_mode(frame_to_scan=None):
        """Active Mode: Listen to user voice query, query Gemini VLM with 5-second temporal memory, and speak response."""
        nonlocal active_mode_active, voice_busy, display_text, display_timer, status_text, status_expiry
        with voice_lock:
            if voice_busy:
                print("  [VOICE] Already processing voice query...", flush=True)
                return
            voice_busy = True

        active_mode_active = True  # Pause YOLO passive scanning
        try:
            print("\n" + "=" * 55, flush=True)
            print("  [ACTIVE MODE] Voice Query to Gemini VLM with Temporal Memory...", flush=True)
            print("=" * 55, flush=True)

            # Interrupt any active passive audio and signal listening mode
            synthesizer.interrupt()
            synthesizer.play_beep()
            time.sleep(0.35)  # Wait for beep to finish so mic doesn't record beep

            with display_lock:
                display_text = "Listening... Ask your question into the microphone"
                display_timer = time.time() + 15.0
            with status_lock:
                status_text = "VOICE: Listening..."
                status_expiry = time.time() + 8.0

            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.pause_threshold = 0.8
            transcription = ""
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio_data = recognizer.listen(source, timeout=8, phrase_time_limit=10)
                    transcription = recognizer.recognize_google(audio_data)
                    print(f"  [VOICE USER]: \"{transcription}\"", flush=True)
            except sr.WaitTimeoutError:
                print("  [VOICE]: No speech detected (timed out).", flush=True)
                synthesizer.speak("I didn't hear anything. Please try again.")
                with display_lock:
                    display_text = "No speech detected. Please try again."
                    display_timer = time.time() + 5.0
                return
            except sr.UnknownValueError:
                print("  [VOICE]: Speech unintelligible.", flush=True)
                synthesizer.speak("I could not understand that. Please try again.")
                with display_lock:
                    display_text = "Could not understand speech."
                    display_timer = time.time() + 5.0
                return
            except Exception as e:
                print(f"  [VOICE ERROR]: Speech recognition error: {e}", flush=True)
                synthesizer.speak("Microphone audio error.")
                with display_lock:
                    display_text = f"Audio error: {e}"
                    display_timer = time.time() + 5.0
                return

            # Check if user asked a location question:
            q_lower = transcription.lower()
            is_location_query = any(phrase in q_lower for phrase in [
                "where am i", "my location", "current location", "where are we",
                "what is my location", "tell me my location", "which street", "what street",
                "what is the location", "what's my location", "my address", "current address"
            ])
            if is_location_query:
                print("  [VOICE ACTION] Location query detected. Resolving via Ola Maps...", flush=True)
                synthesizer.speak("Pinpointing your location.", interrupt=True)
                lat, lng = location_manager.get_coordinates()
                if lat and lng:
                    addr = location_manager.get_street_address(lat, lng)
                    loc_answer = f"You are currently near {addr}."
                else:
                    loc_answer = "Unable to determine your current location."
                print(f"  [LOCATION ANSWER]: {loc_answer}\n", flush=True)
                with display_lock:
                    display_text = f"Q: {transcription}\nA: {loc_answer}"
                    display_timer = time.time() + 12.0
                with status_lock:
                    status_text = f"LOCATION: {loc_answer[:36]}..."
                    status_expiry = time.time() + 8.0
                synthesizer.speak(loc_answer, interrupt=True)
                return

            with display_lock:
                display_text = f"You asked: \"{transcription}\"\nThinking..."
                display_timer = time.time() + 15.0
            with status_lock:
                status_text = "VOICE: Querying Gemini VLM with Temporal Memory..."
                status_expiry = time.time() + 8.0

            # Format payload with the chronological sequence of recent_frames (5-second window)
            temporal_payload = list(recent_frames)
            if not temporal_payload and frame_to_scan is not None:
                temporal_payload = [frame_to_scan]

            synthesizer.speak("Thinking...", interrupt=True)
            answer = ocr_reader.ask_vlm(temporal_payload, transcription)
            print(f"  [GEMINI ANSWER]: {answer}\n", flush=True)

            with display_lock:
                display_text = f"Q: {transcription}\nA: {answer}"
                display_timer = time.time() + 10.0
            with status_lock:
                status_text = f"GEMINI: {answer[:38]}..."
                status_expiry = time.time() + 8.0

            synthesizer.speak(answer, interrupt=True)

        finally:
            active_mode_active = False  # Resume YOLO passive scanning
            with voice_lock:
                voice_busy = False

    # Alias trigger_voice to trigger_active_mode for backward compatibility
    trigger_voice = trigger_active_mode

    def trigger_location():
        """Location Awareness: Get coordinates, reverse geocode via Ola Maps, and announce street address."""
        nonlocal location_busy, display_text, display_timer, status_text, status_expiry
        with location_lock:
            if location_busy:
                print("  [LOCATION] Already fetching location, please wait...", flush=True)
                return
            location_busy = True

        try:
            print("\n" + "=" * 55, flush=True)
            print("  [ACTION] Locating User via Ola Maps...", flush=True)
            print("=" * 55, flush=True)

            with display_lock:
                display_text = "Location: Pinpointing your position..."
                display_timer = time.time() + 10.0
            with status_lock:
                status_text = "LOCATION: Pinpointing position..."
                status_expiry = time.time() + 6.0
            synthesizer.speak("Pinpointing your location.", interrupt=True)

            lat, lng = location_manager.get_coordinates()
            if lat is None or lng is None:
                msg = "Unable to determine your current location."
                print(f"  [LOCATION]: {msg}", flush=True)
                with display_lock:
                    display_text = f"Location:\n{msg}"
                    display_timer = time.time() + 6.0
                synthesizer.speak(msg, interrupt=True)
                return

            print(f"  [COORDINATES]: Latitude {lat:.4f}, Longitude {lng:.4f}", flush=True)
            with status_lock:
                status_text = "LOCATION: Resolving address via Ola Maps..."
                status_expiry = time.time() + 6.0

            address = location_manager.get_street_address(lat, lng)
            print(f"  [OLA MAPS ADDRESS]: {address}\n", flush=True)

            spoken_message = f"You are currently near {address}"
            with display_lock:
                display_text = f"Location:\n{address}"
                display_timer = time.time() + 12.0
            with status_lock:
                status_text = f"LOCATION: {address[:36]}..."
                status_expiry = time.time() + 8.0

            synthesizer.speak(spoken_message, interrupt=True)
        except Exception as e:
            print(f"  [LOCATION ERROR]: {e}", flush=True)
            synthesizer.speak("Could not retrieve your location.", interrupt=True)
            synthesizer.speak("Could not retrieve your location.")
        finally:
            with location_lock:
                location_busy = False

    # ────────────────────────────────────────────────────
    #  Background thread 1: YOLO obstacle detection
    # ────────────────────────────────────────────────────
    def yolo_worker():
        while pipeline_running:
            if active_mode_active:
                time.sleep(0.05)
                continue
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

                # Fuse detected persons with recognized face names
                with face_lock:
                    faces_snapshot = list(cached_faces)

                if faces_snapshot:
                    for d in dets:
                        if d.label.lower() == "person":
                            px1, py1, px2, py2 = d.bbox
                            for face_name, face_box in faces_snapshot:
                                if face_name != "Unknown":
                                    fx1, fy1, fx2, fy2 = face_box
                                    fc_x = (fx1 + fx2) / 2.0
                                    fc_y = (fy1 + fy2) / 2.0
                                    # If face center is in the upper portion of the person bounding box
                                    if px1 <= fc_x <= px2 and (py1 - 40) <= fc_y <= (py1 + (py2 - py1) * 0.65):
                                        d.label = face_name
                                        break

                haz = detector.get_hazards(dets)

                with detection_lock:
                    detections_shared.clear()
                    detections_shared.extend(dets)
                    hazards_shared.clear()
                    hazards_shared.extend(haz)

                # Process hazard alerts through AudioCooldownMatrix & AudioSynthesizer
                for det in haz:
                    alert_matrix.emit_alert(det, synthesizer=synthesizer)
                    # Sync with face_alert_tracker so face announcement doesn't duplicate
                    face_alert_tracker[f"face_{det.label.lower()}"] = time.time()

            except Exception:
                pass
            time.sleep(0.065)  # Cap YOLO inference at ~15 FPS to optimize Pi 5 thermal & CPU headroom

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
    print(f"\n  [Pipeline] {enrolled_count} face(s) enrolled in database.", flush=True)
    print("  [Pipeline] Running. Controls (click video window first):", flush=True)
    print("             Press [R] -> Read Text (Gemini VLM)", flush=True)
    print("             Press [C] -> Check Currency (INR Denomination)", flush=True)
    print("             Press [V] -> Voice Query (Active Scene Ask)", flush=True)
    print("             Press [L] -> Locate User (Ola Maps Reverse Geocoding)", flush=True)
    print("             Press [Q] -> Quit Pipeline\n", flush=True)
    print("  ─── Terminal Alerts ─────────────────────────────", flush=True)

    try:
        reader.start()

        # OS-level global keyboard hooks
        debounce_map = {"time": 0.0}
        def on_key_trigger(handler, name, needs_frame=False):
            def callback(e=None):
                now_t = time.time()
                if now_t - debounce_map["time"] > 0.6:
                    debounce_map["time"] = now_t
                    print(f"\n  [GLOBAL HOOK] Hotkey triggered: [{name}]", flush=True)
                    cur_f = reader.read()
                    if needs_frame and cur_f is None:
                        return
                    if needs_frame:
                        threading.Thread(target=handler, args=(cur_f.copy(),), daemon=True).start()
                    else:
                        threading.Thread(target=handler, daemon=True).start()
            return callback

        try:
            keyboard.on_press_key("r", on_key_trigger(trigger_ocr, "R: OCR Read", needs_frame=True))
            keyboard.on_press_key("c", on_key_trigger(trigger_currency, "C: Currency Scanner", needs_frame=True))
            keyboard.on_press_key("v", on_key_trigger(trigger_voice, "V: Voice Gemini VLM", needs_frame=True))
            keyboard.on_press_key("l", on_key_trigger(trigger_location, "L: Ola Maps Location", needs_frame=False))
            print("  [Pipeline] OS-level global hotkeys active: [R], [C], [V], [L].", flush=True)
        except Exception as _k_err:
            print(f"  [Pipeline Warning] Global hook notice: {_k_err}", flush=True)

        # Launch background daemon workers
        threading.Thread(target=yolo_worker, daemon=True).start()
        threading.Thread(target=face_worker_thread, daemon=True).start()

        # ── Main loop: ONLY display & overlays (zero inference latency) ──
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            current_frame = frame

            # ── VLM Temporal Memory: 5-second rolling window FIFO Ring Buffer ──
            if time.time() - last_save_time >= 1.0:
                small_frame = cv2.resize(current_frame, (640, 480))
                ret, encoded_jpeg = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if ret:
                    recent_frames.append(encoded_jpeg.tobytes())
                last_save_time = time.time()

            h, w = frame.shape[:2]

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

            # Draw base vision overlays
            draw_sector_guides(frame)
            draw_detections(frame, detections)
            draw_face_boxes(frame, current_faces, face_alert_tracker, synthesizer=synthesizer)

            # Draw HUD bar
            now = time.time()
            draw_hud(frame, len(detections), len(hazards), fps)

            # Draw interactive shortcut footer
            cv2.rectangle(frame, (0, h - 22), (w, h), (0, 0, 0), -1)
            footer_text = "Press [R]: Read  |  [C]: Currency  |  [V]: Voice  |  [L]: Location  |  [Q]: Quit"
            with status_lock:
                if now < status_expiry and status_text:
                    footer_text = f">> {status_text} <<"
            cv2.putText(
                frame, footer_text, (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
            )

            # ── Assistive Readout Overlay (Currency, Smart Document Reader, Active Voice & Location) ──
            with display_lock:
                current_display_text = display_text
                current_display_timer = display_timer

            if now < current_display_timer and current_display_text:
                is_currency = current_display_text.startswith("Currency")
                is_voice = "You asked:" in current_display_text or "Listening..." in current_display_text or current_display_text.startswith("Q:")
                is_location = current_display_text.startswith("Location:")
                
                font_scale = 0.75 if (is_currency or is_voice) else 0.55
                font_thick = 2 if (is_currency or is_voice) else 1
                line_spacing = 32 if (is_currency or is_voice) else 24
                box_padding = 14
                max_w = max(200, w - 60)

                # Word-wrap multi-line/paragraph text so it never overflows off-screen
                wrapped_lines = wrap_text_to_lines(
                    current_display_text, max_pixel_width=max_w, font_scale=font_scale, thickness=font_thick
                )
                
                display_lines = wrapped_lines[:10]
                total_box_h = box_padding * 2 + len(display_lines) * line_spacing + 24

                # Position near top (y=46, right below the 40px HUD) so it is NEVER cut off by the Windows taskbar
                box_y1 = 46
                box_y2 = min(h - 30, box_y1 + total_box_h)

                # High-contrast solid dark rectangle card
                cv2.rectangle(frame, (10, box_y1), (w - 10, box_y2), (15, 15, 15), -1)

                # Accent border (Gold for Currency, Violet/Magenta for Voice, Amber for Location, Cyan for OCR)
                if is_currency:
                    border_color = (0, 215, 255)
                    header = "ASSISTIVE READOUT (Currency Check):"
                    text_color = (0, 235, 255)
                elif is_voice:
                    border_color = (255, 100, 220)
                    header = "ACTIVE VOICE MODE (Gemini VLM):"
                    text_color = (255, 255, 255)
                elif is_location:
                    border_color = (0, 165, 255)
                    header = "ASSISTIVE READOUT (Ola Maps Location):"
                    text_color = (0, 220, 255)
                else:
                    border_color = (0, 255, 200)
                    header = "ASSISTIVE READOUT (Smart Document Reader):"
                    text_color = (255, 255, 255)

                cv2.rectangle(frame, (10, box_y1), (w - 10, box_y2), border_color, 2)

                # Header badge
                cv2.putText(
                    frame, header, (24, box_y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_color, 1, cv2.LINE_AA
                )

                # Render text lines
                for idx, line_str in enumerate(display_lines):
                    line_y = box_y1 + 24 + box_padding + idx * line_spacing + 12
                    cv2.putText(
                        frame, line_str, (24, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA
                    )

            window_title = "Saarthi — Hazard Perception"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.imshow(window_title, frame)

            cv_key = cv2.waitKey(1) & 0xFF

            # ── Global Keyboard Listener (works even when browser/dashboard is focused) ──
            now_time = time.time()
            if now_time - last_key_time > 0.45:
                try:
                    if keyboard.is_pressed("q") or cv_key in (ord("q"), ord("Q")):
                        break
                    elif keyboard.is_pressed("r") or cv_key in (ord("r"), ord("R")):
                        last_key_time = now_time
                        print("\n  [GLOBAL INPUT] 'R' pressed -> Capturing frame for OCR...", flush=True)
                        with status_lock:
                            status_text = "OCR: Reading text... Please hold steady"
                            status_expiry = time.time() + 6.0
                        threading.Thread(target=trigger_ocr, args=(frame.copy(),), daemon=True).start()
                    elif keyboard.is_pressed("c") or cv_key in (ord("c"), ord("C")):
                        last_key_time = now_time
                        print("\n  [GLOBAL INPUT] 'C' pressed -> Capturing frame for Currency check...", flush=True)
                        with status_lock:
                            status_text = "CURRENCY: Scanning note... Please hold steady"
                            status_expiry = time.time() + 6.0
                        threading.Thread(target=trigger_currency, args=(frame.copy(),), daemon=True).start()
                    elif keyboard.is_pressed("v") or cv_key in (ord("v"), ord("V")):
                        last_key_time = now_time
                        print("\n  [GLOBAL INPUT] 'V' pressed -> Starting Voice/Active Mode...", flush=True)
                        threading.Thread(target=trigger_voice, args=(frame.copy(),), daemon=True).start()
                    elif keyboard.is_pressed("l") or cv_key in (ord("l"), ord("L")):
                        last_key_time = now_time
                        print("\n  [GLOBAL INPUT] 'L' pressed -> Fetching current location via Ola Maps...", flush=True)
                        with status_lock:
                            status_text = "LOCATION: Fetching location..."
                            status_expiry = time.time() + 6.0
                        threading.Thread(target=trigger_location, daemon=True).start()
                except Exception:
                    pass

    except ConnectionError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  [Pipeline] Interrupted by user.")
    finally:
        pipeline_running = False
        reader.stop()
        synthesizer.stop()
        cv2.destroyAllWindows()
        print("  [Pipeline] Clean shutdown complete.\n")


if __name__ == "__main__":
    main()
