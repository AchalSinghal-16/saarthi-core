"""
Saarthi -- Caregiver Web Dashboard & Telemetry Server
=====================================================
FastAPI master application entrypoint.
Wraps the Saarthi edge perception pipeline (YOLOv8 + FaceNet + ChromaDB)
with real-time multipart JPEG video telemetry, biometric face administration,
and caregiver remote configuration.
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any

# Ensure backend root is in sys.path
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BACKEND_DIR)

import cv2
import numpy as np
import psutil
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from vision.stream_reader import ThreadedStreamReader
from vision.obstacle_detector import ObstacleDetector, HAZARD_ZONE_M
from vision.face_memory import FaceMemory
from vision.ocr_reader import OCRReader
from vision.currency_detector import CurrencyDetector
from audio.synthesizer import AudioSynthesizer
from navigation.locator import LocationManager
import keyboard
import speech_recognition as sr
import imageio_ffmpeg
from pydub import AudioSegment

# Link ffmpeg binary for pydub conversion
try:
    _ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_bin)
    _ffmpeg_copy = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(_ffmpeg_copy):
        import shutil
        shutil.copyfile(_ffmpeg_bin, _ffmpeg_copy)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_ffmpeg_dir};{os.environ.get('PATH', '')}"
    AudioSegment.converter = _ffmpeg_copy
except Exception as _fe:
    pass

# ──────────────────────────────────────────────────────────────
#  Application & Configuration
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Saarthi Telemetry & Caregiver Server",
    description="Live video telemetry feed, biometric face enrollment, and caregiver configuration.",
    version="1.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Emergency Configuration & Persistence
SETTINGS_FILE = os.path.join(_BACKEND_DIR, "settings.json")

def load_emergency_settings() -> dict:
    """Load persisted emergency contact settings from disk, or fallback to default."""
    default_settings = {
        "phone_number": "+91 98765 43210",
        "sos_message": "EMERGENCY: User requires immediate assistance at their current location.",
        "twilio_configured": True,
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict) and "phone_number" in saved:
                    return {**default_settings, **saved}
        except Exception as e:
            print(f"[Settings Warning] Could not load {SETTINGS_FILE}: {e}")
    return default_settings

def save_emergency_settings(settings: dict):
    """Save emergency configuration permanently to disk."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        print(f"[Settings] Saved emergency configuration to {SETTINGS_FILE}")
    except Exception as e:
        print(f"[Settings Error] Could not save {SETTINGS_FILE}: {e}")

EMERGENCY_SETTINGS = load_emergency_settings()


class EmergencySettingsPayload(BaseModel):
    phone_number: str
    sos_message: Optional[str] = None


class VideoSourcePayload(BaseModel):
    source: str


class LocationCoordinatesPayload(BaseModel):
    lat: float
    lng: float
    source: Optional[str] = "Browser GPS"


class TriggerLocationPayload(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None


# ──────────────────────────────────────────────────────────────
#  Perception Pipeline Shared State & Workers
# ──────────────────────────────────────────────────────────────

VIDEO_SOURCE = os.getenv("SAARTHI_VIDEO_SOURCE", "0")
for i, arg in enumerate(sys.argv):
    if arg == "--source" and i + 1 < len(sys.argv):
        VIDEO_SOURCE = sys.argv[i + 1]
        break

if str(VIDEO_SOURCE).isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)

# Initialize core edge components
print("=" * 60)
print("  Saarthi -- Initializing Telemetry Server Components...")
print("=" * 60)

detector = ObstacleDetector()
face_memory = FaceMemory()
synthesizer = AudioSynthesizer()
location_manager = LocationManager()
ocr_reader = OCRReader()
currency_detector = CurrencyDetector(ocr_reader=ocr_reader)
reader: Optional[ThreadedStreamReader] = None

# Thread-safe pipeline caches
detection_lock = threading.Lock()
detections_shared = []
hazards_shared = []

face_lock = threading.Lock()
cached_faces = []

action_lock = threading.Lock()
action_busy = False

pipeline_running = True
prev_time = time.time()
camera_lock = threading.Lock()

# VLM Temporal Memory: 5-second rolling window FIFO ring buffer
recent_frames = deque(maxlen=5)
last_save_time = time.time()

# ──────────────────────────────────────────────────────────────
#  Assistive Actions (Callable via Global Hotkeys or REST API)
# ──────────────────────────────────────────────────────────────

def trigger_ocr():
    global action_busy
    with action_lock:
        if action_busy:
            return "Busy processing an action."
        action_busy = True
    try:
        if reader is None:
            with action_lock:
                action_busy = False
            return "Camera stream not connected."
        frame = reader.read()
        if frame is None:
            with action_lock:
                action_busy = False
            return "No camera frame available."

        print("\n  [ACTION] Processing Document OCR with Gemini VLM in Background Thread...", flush=True)
        synthesizer.speak("Reading text, please hold steady.", interrupt=True)

        def _async_ocr_worker(frame_img):
            global action_busy
            try:
                text_out = ocr_reader.read_text(frame_img)
                if text_out:
                    print(f"  [OCR TEXT]: {text_out}", flush=True)
                    # Automatically speak the text returned from Gemini
                    synthesizer.speak(text_out, interrupt=True)
                else:
                    print("  [OCR TEXT]: No legible text detected.", flush=True)
                    synthesizer.speak("No legible text detected.", interrupt=True)
            except Exception as err:
                print(f"  [OCR ERROR]: {err}", flush=True)
                synthesizer.speak("Could not read text.", interrupt=True)
            finally:
                with action_lock:
                    action_busy = False

        # Launch heavy Gemini API network call on background thread
        threading.Thread(target=_async_ocr_worker, args=(frame.copy(),), daemon=True).start()
        return "Reading document text with Gemini VLM..."
    except Exception as e:
        with action_lock:
            action_busy = False
        print(f"  [OCR ERROR]: {e}", flush=True)
        return str(e)

def trigger_currency():
    global action_busy
    with action_lock:
        if action_busy:
            return "Busy processing an action."
        action_busy = True
    try:
        if reader is None:
            return "Camera stream not connected."
        frame = reader.read()
        if frame is None:
            return "No camera frame available."
        print("\n  [ACTION] Scanning Indian Rupee Currency with Gemini VLM...", flush=True)
        synthesizer.speak("Scanning note, please hold steady.", interrupt=True)

        denom, conf = currency_detector.detect_currency(frame)
        announcement = f"Currency detected: {denom}" if conf > 0 else denom
        print(f"  [CURRENCY RESULT]: {announcement}", flush=True)
        synthesizer.speak(announcement, interrupt=True)
        return announcement
    except Exception as e:
        print(f"  [CURRENCY ERROR]: {e}", flush=True)
        synthesizer.speak("Could not identify currency note.", interrupt=True)
        return str(e)
    finally:
        with action_lock:
            action_busy = False

def trigger_location(lat: Optional[float] = None, lng: Optional[float] = None):
    global action_busy
    with action_lock:
        if action_busy:
            return "Busy processing an action."
        action_busy = True
    try:
        print("\n  [ACTION] Resolving Location via Ola Maps...", flush=True)
        if lat is not None and lng is not None:
            location_manager.update_coordinates(lat, lng, source="Browser GPS")

        synthesizer.speak("Pinpointing your location.", interrupt=True)
        resolved_lat, resolved_lng = location_manager.get_coordinates()
        if resolved_lat is None or resolved_lng is None:
            synthesizer.speak("Location unavailable.", interrupt=True)
            return "Location unavailable."
        addr = location_manager.get_street_address(resolved_lat, resolved_lng)
        announcement = f"You are currently near {addr}"
        print(f"  [LOCATION]: {announcement}", flush=True)
        synthesizer.speak(announcement, interrupt=True)
        return announcement
    except Exception as e:
        print(f"  [LOCATION ERROR]: {e}", flush=True)
        synthesizer.speak("Could not retrieve your location.", interrupt=True)
        return str(e)
    finally:
        with action_lock:
            action_busy = False

def process_audio_file_transcription(audio_bytes: bytes) -> Optional[str]:
    """Converts uploaded audio bytes (webm/wav) to wav and transcribes via Google Speech Recognition."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_in = os.path.join(backend_dir, f"temp_upload_{int(time.time() * 1000)}.webm")
    tmp_wav = os.path.join(backend_dir, f"temp_upload_{int(time.time() * 1000)}.wav")
    try:
        with open(tmp_in, "wb") as f:
            f.write(audio_bytes)
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_bin, "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", tmp_wav]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(tmp_wav):
            print(f"[Voice Upload Error] ffmpeg conversion failed: {res.stderr}", flush=True)
            return None
        r = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("[Voice Upload] Speech unintelligible.", flush=True)
        return None
    except Exception as e:
        print(f"[Voice Upload Error]: {e}", flush=True)
        return None
    finally:
        for p in (tmp_in, tmp_wav):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def trigger_voice(question: Optional[str] = None, audio_bytes: Optional[bytes] = None):
    global action_busy
    with action_lock:
        if action_busy:
            return "Busy processing an action."
        action_busy = True
    try:
        print("\n  [ACTION] Active Voice Mode (Gemini VLM & Location)...", flush=True)
        synthesizer.interrupt()

        resolved_question = (question or "").strip()

        # 1. If audio file was uploaded from browser or phone:
        if not resolved_question and audio_bytes and len(audio_bytes) > 200:
            print("  [VOICE] Transcribing uploaded audio recording...", flush=True)
            resolved_question = process_audio_file_transcription(audio_bytes)
            if not resolved_question:
                synthesizer.speak("I could not understand that. Please try again.", interrupt=True)
                return "Could not understand speech from uploaded audio."

        # 2. If neither question nor audio was provided (local hardware hotkey):
        if not resolved_question:
            synthesizer.play_beep()
            time.sleep(0.35)  # Allow beep to finish before mic recording

            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.pause_threshold = 0.8
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio_data = recognizer.listen(source, timeout=8, phrase_time_limit=10)
                    resolved_question = recognizer.recognize_google(audio_data)
                    print(f"  [VOICE USER]: \"{resolved_question}\"", flush=True)
            except sr.WaitTimeoutError:
                print("  [VOICE]: No speech detected (timed out).", flush=True)
                synthesizer.speak("I didn't hear anything. Please try again.", interrupt=True)
                return "No speech detected. Please speak clearly into the microphone."
            except sr.UnknownValueError:
                print("  [VOICE]: Speech unintelligible.", flush=True)
                synthesizer.speak("I could not understand that. Please try again.", interrupt=True)
                return "Could not understand speech. Please try again."

        if not resolved_question:
            return "No voice question captured."

        # Check if user asked a location question:
        q_lower = resolved_question.lower()
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
            print(f"  [LOCATION ANSWER]: {loc_answer}", flush=True)
            synthesizer.speak(loc_answer, interrupt=True)
            return f"Q: {resolved_question} | A: {loc_answer}"

        # Visual question requires camera frame or temporal ring buffer
        temporal_frames = list(recent_frames)
        if not temporal_frames:
            frame = reader.read() if reader is not None else None
            if frame is None:
                msg = "Camera stream not connected. Please ensure camera is active."
                synthesizer.speak(msg, interrupt=True)
                return msg
            temporal_frames = [frame]

        print(f"  [GEMINI QUERY WITH TEMPORAL MEMORY ({len(temporal_frames)} frames)]: \"{resolved_question}\"", flush=True)
        synthesizer.speak("Thinking...", interrupt=True)
        answer = ocr_reader.ask_vlm(temporal_frames, resolved_question)
        print(f"  [GEMINI ANSWER]: {answer}", flush=True)
        synthesizer.speak(answer, interrupt=True)
        return f"Q: {resolved_question} | A: {answer}"
    except Exception as e:
        print(f"  [VOICE ERROR]: {e}", flush=True)
        synthesizer.speak("Could not complete voice query.")
        return str(e)
    finally:
        with action_lock:
            action_busy = False

# Alias trigger_active_mode to trigger_voice
trigger_active_mode = trigger_voice

def setup_global_keyboard_hooks():
    """Register OS-level asynchronous keyboard hooks so hotkeys fire globally."""
    debounce = {"time": 0.0}
    def wrap_hook(fn, label):
        def handler(e=None):
            now = time.time()
            if now - debounce["time"] > 0.6:
                debounce["time"] = now
                print(f"\n[GLOBAL KEYBOARD] Hotkey triggered: [{label}]", flush=True)
                threading.Thread(target=fn, daemon=True).start()
        return handler

    try:
        keyboard.on_press_key("r", wrap_hook(trigger_ocr, "R: Read Document"))
        keyboard.on_press_key("c", wrap_hook(trigger_currency, "C: Currency Scanner"))
        keyboard.on_press_key("l", wrap_hook(trigger_location, "L: Location Announcement"))
        keyboard.on_press_key("v", wrap_hook(trigger_voice, "V: Voice Ask Gemini"))
        print("[GlobalKeyboard] OS-level keyboard hooks active for keys [R, C, L, V].", flush=True)
    except Exception as err:
        print(f"[GlobalKeyboard Warning] OS hook could not be established: {err}", flush=True)

# ──────────────────────────────────────────────────────────────
#  Alert Fatigue & Audio Dispatcher
# ──────────────────────────────────────────────────────────────

class AudioCooldownMatrix:
    """
    Suppresses repetitive hazard warnings to prevent alert fatigue.
    Default cooldown of 5.0 seconds per (label, sector).
    Delta-distance override if obstacle moves closer by >= 0.5m.
    """
    COOLDOWN_SECONDS = 5.0
    DELTA_DISTANCE_M = 0.5

    def __init__(self, cooldown_seconds: float = 5.0, delta_distance_m: float = 0.5):
        self.cooldown_seconds = cooldown_seconds
        self.delta_distance_m = delta_distance_m
        self.last_alert_times = {}

    def should_alert(self, det) -> bool:
        now = time.time()
        key = (det.label.lower(), det.direction)
        if key not in self.last_alert_times:
            self.last_alert_times[key] = {"timestamp": now, "distance": det.distance_m}
            return True

        record = self.last_alert_times[key]
        time_elapsed = now - record["timestamp"]
        prev_distance = record["distance"]

        is_closer_by_delta = (
            prev_distance > 0
            and det.distance_m > 0
            and (prev_distance - det.distance_m >= self.delta_distance_m)
        )

        if is_closer_by_delta or (time_elapsed >= self.cooldown_seconds):
            self.last_alert_times[key] = {"timestamp": now, "distance": det.distance_m}
            return True

        return False

    def emit_alert(self, det, synthesizer=None):
        if self.should_alert(det):
            dist_str = f"{det.distance_m:.1f}m" if det.distance_m > 0 else "nearby"
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"  [{timestamp}] [AUDIO ALERT] {det.label.capitalize()} {dist_str} in {det.direction}", flush=True)
            if synthesizer is not None:
                synthesizer.speak(f"{det.label}, {det.direction}")


alert_matrix = AudioCooldownMatrix(cooldown_seconds=5.0, delta_distance_m=0.5)
face_alert_tracker = {}


def init_camera(source):
    """Safely start or switch the video stream source."""
    global reader, VIDEO_SOURCE
    with camera_lock:
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
            reader = None

        VIDEO_SOURCE = int(source) if str(source).isdigit() else str(source).strip()
        try:
            r = ThreadedStreamReader(source=VIDEO_SOURCE)
            r.start()
            reader = r
            print(f"[TelemetryServer] Video stream established on: {VIDEO_SOURCE}", flush=True)
            if isinstance(VIDEO_SOURCE, str) and "://" in VIDEO_SOURCE:
                location_manager.set_sensor_source(VIDEO_SOURCE)
            return True, f"Connected to {VIDEO_SOURCE}"
        except Exception as e:
            err_msg = str(e)
            print(f"[TelemetryServer Warning] Could not start source {VIDEO_SOURCE}: {err_msg}", flush=True)
            reader = None
            return False, err_msg


# ──────────────────────────────────────────────────────────────
#  Drawing helpers for telemetry overlay
# ──────────────────────────────────────────────────────────────

def draw_sector_guides(frame: np.ndarray):
    """Draw faint vertical sector guide lines on telemetry frame."""
    h, w = frame.shape[:2]
    left_x = int(w * 0.35)
    right_x = int(w * 0.65)
    color = (80, 80, 80)
    cv2.line(frame, (left_x, 0), (left_x, h), color, 1)
    cv2.line(frame, (right_x, 0), (right_x, h), color, 1)
    cv2.putText(frame, "LEFT", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, "CENTER", (left_x + 10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, "RIGHT", (right_x + 10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_detections(frame: np.ndarray, detections: list):
    """Draw bounding boxes and distance tags on telemetry frame."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = (0, 0, 255) if det.is_hazard else (0, 220, 0)
        thickness = 3 if det.is_hazard else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        dist_str = f"{det.distance_m:.1f}m" if det.distance_m > 0 else "?m"
        label_text = f"{det.label.capitalize()} {dist_str} [{det.direction}]"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_face_boxes(frame: np.ndarray, detected_faces: list):
    """Render face bounding boxes and recognized names."""
    for name, bbox in detected_faces:
        x1, y1, x2, y2 = bbox
        is_known = name != "Unknown"
        color = (0, 255, 0) if is_known else (0, 165, 255)
        text_color = (0, 0, 0) if is_known else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"Face: {name}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)


def draw_hud(frame: np.ndarray, total: int, hazard_count: int, fps: float):
    """Render telemetry header HUD."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (15, 15, 15), -1)
    hud_color = (0, 0, 255) if hazard_count > 0 else (0, 255, 120)
    hud_text = f"SAARTHI TELEMETRY  |  Objects: {total}  |  Hazards: {hazard_count}  |  FPS: {fps:.0f}"
    cv2.putText(frame, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, hud_color, 2)


# ──────────────────────────────────────────────────────────────
#  Background processing workers
# ──────────────────────────────────────────────────────────────

def yolo_worker():
    """Background thread running YOLOv8 inference and spatial hazard detection."""
    global detections_shared, hazards_shared, last_save_time, recent_frames
    while pipeline_running:
        if reader is None:
            time.sleep(0.1)
            continue
        frame = reader.read()
        if frame is None:
            time.sleep(0.02)
            continue
        try:
            # ── VLM Temporal Memory: 5-second rolling window FIFO Ring Buffer ──
            if time.time() - last_save_time >= 1.0:
                small_frame = cv2.resize(frame, (640, 480))
                ret, encoded_jpeg = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                if ret:
                    recent_frames.append(encoded_jpeg.tobytes())
                last_save_time = time.time()
            small = cv2.resize(frame, (640, 480))
            dets = detector.detect(small)

            h_orig, w_orig = frame.shape[:2]
            sx, sy = w_orig / 640.0, h_orig / 480.0
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                d.bbox = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

            # Fuse recognized face identities into person detections
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
                                if px1 <= fc_x <= px2 and (py1 - 40) <= fc_y <= (py1 + (py2 - py1) * 0.65):
                                    d.label = face_name
                                    break

            haz = detector.get_hazards(dets)

            with detection_lock:
                detections_shared = list(dets)
                hazards_shared = list(haz)

            # Route spatial hazard alerts to audio synthesizer
            for det in haz:
                alert_matrix.emit_alert(det, synthesizer=synthesizer)
                face_alert_tracker[f"face_{det.label.lower()}"] = time.time()

        except Exception:
            pass
        time.sleep(0.065)  # Cap YOLO inference at ~15 FPS to optimize Pi 5 thermal & CPU headroom


def face_worker():
    """Background thread running FaceNet facial identification."""
    global cached_faces
    while pipeline_running:
        if reader is None:
            time.sleep(0.2)
            continue
        frame = reader.read()
        if frame is not None:
            try:
                results = face_memory.identify_face(frame)
                with face_lock:
                    cached_faces = list(results)

                # Announce recognized familiar contacts aloud
                for face_name, bbox in results:
                    if face_name != "Unknown":
                        now = time.time()
                        key = f"face_{face_name.lower()}"
                        if key not in face_alert_tracker or (now - face_alert_tracker[key] >= 12.0):
                            face_alert_tracker[key] = now
                            fx1, fy1, fx2, fy2 = bbox
                            frame_w = frame.shape[1]
                            fc_x = (fx1 + fx2) / 2.0
                            direction = "center"
                            if fc_x < frame_w * 0.35:
                                direction = "left"
                            elif fc_x > frame_w * 0.65:
                                direction = "right"
                            print(f"  [AUDIO ALERT] Recognized {face_name} in {direction}", flush=True)
                            synthesizer.speak(f"{face_name}, {direction}")
            except Exception:
                pass
        time.sleep(1.0)


# ──────────────────────────────────────────────────────────────
#  FastAPI Startup & Shutdown Lifespan
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    """Initialize camera stream and background workers upon server boot."""
    print(f"\n[TelemetryServer] Starting perception pipeline (source: {VIDEO_SOURCE})...", flush=True)
    init_camera(VIDEO_SOURCE)
    threading.Thread(target=yolo_worker, daemon=True).start()
    threading.Thread(target=face_worker, daemon=True).start()
    setup_global_keyboard_hooks()
    synthesizer.speak("Saarthi online. Vision perception active.")
    print("[TelemetryServer] Pipeline initialized and ready.\n", flush=True)


@app.on_event("shutdown")
def on_shutdown():
    global pipeline_running, reader
    pipeline_running = False
    if reader is not None:
        try:
            reader.stop()
        except Exception:
            pass
    synthesizer.stop()


# ──────────────────────────────────────────────────────────────
#  Step 2: Video Stream Telemetry Generator
# ──────────────────────────────────────────────────────────────

async def video_stream():
    """
    Asynchronous generator yielding JPEG-encoded frames in multipart/x-mixed-replace format.
    Renders live YOLO bounding boxes or an informative standby card if the camera is connecting.
    """
    global prev_time
    tick = 0
    while pipeline_running:
        frame = reader.read() if reader is not None else None

        if frame is None:
            tick += 1
            # Standby slate card
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (15, 23, 42)  # Dark slate background

            # Pulsing status indicator
            dot_color = (0, 215, 255) if (tick % 20 < 10) else (70, 85, 110)
            cv2.circle(placeholder, (65, 200), 7, dot_color, -1)

            cv2.putText(
                placeholder, "Connecting to Saarthi Camera Stream...",
                (88, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                placeholder, f"Target Source: {VIDEO_SOURCE}",
                (65, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                placeholder, "Start IP Webcam on phone or enter stream URL in Dashboard",
                (65, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA
            )

            ret, buffer = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            await asyncio.sleep(0.08)
            continue

        # Create overlay copy for client telemetry
        display_frame = frame.copy()

        with detection_lock:
            cur_dets = list(detections_shared)
            cur_haz = list(hazards_shared)

        with face_lock:
            cur_faces = list(cached_faces)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Render overlays
        draw_sector_guides(display_frame)
        draw_detections(display_frame, cur_dets)
        draw_face_boxes(display_frame, cur_faces)
        draw_hud(display_frame, len(cur_dets), len(cur_haz), fps)

        # Encode to JPEG
        ret, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            await asyncio.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        await asyncio.sleep(0.066)  # Cap telemetry stream at ~15 FPS to prevent Pi Wi-Fi/thermal congestion


@app.get("/api/video_feed")
def get_video_feed():
    """
    Live video telemetry stream for the caregiver dashboard.
    Returns: multipart/x-mixed-replace image/jpeg stream.
    """
    return StreamingResponse(
        video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ──────────────────────────────────────────────────────────────
#  Step 3: Biometric & Configuration API Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/faces/enroll")
async def enroll_face(file: UploadFile = File(...), name: str = Form(...)):
    """
    Upload a portrait photo to enroll a known face into ChromaDB.
    """
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Invalid image format. Please upload JPEG or PNG."}
            )

        face_id = face_memory.enroll(img, clean_name)

        if face_id:
            return {
                "success": True,
                "message": f"Successfully enrolled {clean_name}.",
                "face_id": face_id,
                "total_faces": face_memory.count(),
            }
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No face detected in the photo. Please upload a clear, front-facing portrait."
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Enrollment error: {str(e)}"}
        )


@app.get("/api/faces")
def get_enrolled_faces():
    """
    Retrieve all enrolled faces stored in the ChromaDB vector database.
    """
    faces = face_memory.list_faces()
    return {
        "success": True,
        "count": len(faces),
        "faces": faces,
    }


@app.post("/api/settings/emergency")
def update_emergency_settings(payload: EmergencySettingsPayload):
    """
    Update caregiver SOS emergency contact telephone number.
    """
    global EMERGENCY_SETTINGS
    clean_phone = payload.phone_number.strip()
    if not clean_phone:
        raise HTTPException(status_code=400, detail="Phone number cannot be empty.")

    EMERGENCY_SETTINGS["phone_number"] = clean_phone
    if payload.sos_message:
        EMERGENCY_SETTINGS["sos_message"] = payload.sos_message.strip()

    save_emergency_settings(EMERGENCY_SETTINGS)

    print(f"[TelemetryServer] Emergency SOS updated and saved: {EMERGENCY_SETTINGS['phone_number']}")
    return {
        "success": True,
        "message": "Emergency contact configuration updated successfully.",
        "settings": EMERGENCY_SETTINGS,
    }


@app.get("/api/settings/emergency")
def get_emergency_settings():
    """
    Retrieve current emergency configuration.
    """
    return {
        "success": True,
        "settings": EMERGENCY_SETTINGS,
    }


@app.post("/api/settings/source")
def update_video_source(payload: VideoSourcePayload):
    """
    Dynamically switch camera input source (e.g. phone IP webcam URL or integer webcam index).
    """
    raw_src = payload.source.strip()
    success, msg = init_camera(raw_src)
    if success:
        return {"success": True, "message": f"Connected to camera stream: {raw_src}", "source": str(raw_src)}
    else:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Could not connect to {raw_src}: {msg}"})


@app.get("/api/settings/source")
def get_video_source():
    """
    Get the active video stream source.
    """
    return {
        "success": True,
        "source": str(VIDEO_SOURCE),
        "connected": reader is not None and getattr(reader, "running", False),
    }


@app.get("/api/location")
def get_location():
    """
    Get the wearable's raw GPS/sensor coordinates and tracking source.
    """
    lat, lng = location_manager.get_coordinates()
    if lat is None or lng is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": "Coordinates could not be determined."}
        )
    return {
        "success": True,
        "lat": lat,
        "lng": lng,
        "coordinates": {"lat": lat, "lng": lng},
        "source": getattr(location_manager, "synced_source", "IP Fallback") or "IP Fallback",
    }


@app.post("/api/location/coordinates")
def update_coordinates_endpoint(payload: LocationCoordinatesPayload):
    """
    Synchronize real-time device coordinates (from browser navigator.geolocation or caregiver map).
    """
    location_manager.update_coordinates(payload.lat, payload.lng, source=payload.source or "Browser GPS")
    addr = location_manager.get_street_address(payload.lat, payload.lng)
    return {
        "success": True,
        "message": "Coordinates updated successfully.",
        "lat": payload.lat,
        "lng": payload.lng,
        "address": addr,
        "source": payload.source or "Browser GPS",
    }


@app.get("/api/vitals")
def get_vitals():
    """
    Telemetry endpoint providing CPU utilization and thermal telemetry.
    """
    cpu_percent = psutil.cpu_percent(interval=None)

    # Read hardware temperature on Raspberry Pi if available, else fallback to mock 45.0°C
    cpu_temp = 45.0
    try:
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            with open(thermal_path, "r") as f:
                cpu_temp = round(float(f.read().strip()) / 1000.0, 1)
        elif hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        cpu_temp = round(entries[0].current, 1)
                        break
    except Exception:
        cpu_temp = 45.0

    return {
        "success": True,
        "cpu_percent": cpu_percent,
        "cpu_temp": cpu_temp,
    }


@app.post("/api/intercom")
async def intercom_message(file: UploadFile = File(...)):
    """
    Caregiver Intercom endpoint.
    Receives WebM audio upload, converts to clean 24kHz Mono WAV via FFmpeg, and plays aloud.
    """
    content = await file.read()
    if not content or len(content) < 100:
        raise HTTPException(status_code=400, detail="Empty or invalid audio recording received.")

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    ts = int(time.time() * 1000)
    webm_path = os.path.join(backend_dir, f"intercom_{ts}.webm")
    wav_path = os.path.join(backend_dir, f"intercom_{ts}.wav")

    with open(webm_path, "wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    playback_file = None
    # Convert .webm to .wav using ffmpeg (via imageio_ffmpeg)
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_bin, "-y", "-i", webm_path, "-ar", "24000", "-ac", "1", wav_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            playback_file = wav_path
            print(f"[Intercom] Converted audio to {wav_path} ({os.path.getsize(wav_path)} bytes).", flush=True)
        else:
            print(f"[Intercom Warning] ffmpeg conversion notice: {res.stderr}", flush=True)
    except Exception as conv_err:
        print(f"[Intercom Warning] ffmpeg conversion exception: {conv_err}", flush=True)

    if not playback_file:
        playback_file = webm_path

    print(f"[Intercom] Playing audio transmission ({len(content)} bytes via {playback_file})...", flush=True)
    played = synthesizer.play_file(playback_file, interrupt=True)

    # Clean up temporary webm input file
    try:
        if os.path.exists(webm_path):
            os.remove(webm_path)
    except Exception:
        pass

    return {
        "success": played,
        "message": f"Intercom transmission processed ({len(content)} bytes) and played.",
    }


@app.post("/api/trigger/read")
def api_trigger_read():
    """Trigger Document OCR & Gemini VLM reading."""
    res = trigger_ocr()
    return {"success": True, "result": res}

@app.post("/api/trigger/currency")
def api_trigger_currency():
    """Trigger Indian Currency scanner."""
    res = trigger_currency()
    return {"success": True, "result": res}

@app.post("/api/trigger/location")
def api_trigger_location(payload: Optional[TriggerLocationPayload] = None):
    """Trigger Ola Maps location announcement with optional client GPS coordinates."""
    lat = payload.lat if payload else None
    lng = payload.lng if payload else None
    res = trigger_location(lat=lat, lng=lng)
    return {"success": True, "result": res}

@app.post("/api/trigger/voice")
@app.post("/api/trigger/active_mode")
async def api_trigger_voice(
    file: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None)
):
    """
    Trigger active voice query to Gemini VLM.
    Accepts:
      - Uploaded audio recording (.webm/.wav) from browser or phone
      - Direct text question (from speech recognition or user input)
      - No params: falls back to local machine microphone via sr.Microphone()
    """
    audio_bytes = await file.read() if file is not None else None
    res = trigger_voice(question=question, audio_bytes=audio_bytes)
    return {"success": True, "result": res}


@app.post("/api/audio/stop")
def api_audio_stop():
    """
    Caregiver Remote Kill-Switch: Immediately halts any currently playing audio on wearable.
    """
    synthesizer.stop()
    return {"success": True, "message": "Audio stopped."}


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Saarthi Edge AI Assistive Wearable",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "video_feed": "/api/video_feed",
            "faces": "/api/faces",
            "enroll": "/api/faces/enroll",
            "emergency": "/api/settings/emergency",
            "source": "/api/settings/source",
        },
    }


# ──────────────────────────────────────────────────────────────
#  Application Server Entrypoint
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Saarthi Telemetry Server")
    parser.add_argument("--source", type=str, default=str(VIDEO_SOURCE), help="Video source (e.g. http://<PHONE_IP>:8080/video or 0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    args, _ = parser.parse_known_args()

    active_src = int(args.source) if args.source.isdigit() else args.source
    VIDEO_SOURCE = active_src

    print(f"\n[TelemetryServer] Starting Uvicorn on http://{args.host}:{args.port} (source: {VIDEO_SOURCE})...", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
