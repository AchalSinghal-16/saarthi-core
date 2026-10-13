"""
Saarthi — Indian Currency Recognition Module
=============================================
Identifies Indian Rupee (INR) banknote denominations:
₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

Architecture:
1. Primary: Custom YOLOv8 weights (models/currency_yolov8n.pt) if present.
2. Fallback: Multi-modal fusion of HSV Color-Histogram profile + OCR Numeral/Watermark
   extraction (e.g., '₹', '10', '20', '50', '100', '200', '500', 'RBI').
"""

import os
import sys
import re
from typing import Tuple, Optional
import cv2
import numpy as np

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CurrencyDetector:
    """
    Detects Indian Rupee banknote denominations in video frames.
    """

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "currency_yolov8n.pt"
    )

    DENOMINATION_NAMES = {
        "10": "10 Rupees",
        "20": "20 Rupees",
        "50": "50 Rupees",
        "100": "100 Rupees",
        "200": "200 Rupees",
        "500": "500 Rupees",
    }

    # HSV Color profiles for Mahatma Gandhi New Series banknotes
    # Format: (Hue_min, Hue_max, Sat_min, Sat_max, Val_min, Val_max)
    COLOR_PROFILES = {
        "10": {   # Chocolate Brown
            "h_range": (8, 22),
            "s_range": (60, 220),
            "v_range": (50, 180),
            "name": "Chocolate Brown",
        },
        "20": {   # Greenish Yellow
            "h_range": (28, 48),
            "s_range": (80, 255),
            "v_range": (100, 255),
            "name": "Greenish Yellow",
        },
        "50": {   # Fluorescent Blue / Cyan
            "h_range": (85, 115),
            "s_range": (90, 255),
            "v_range": (90, 255),
            "name": "Fluorescent Blue",
        },
        "100": {  # Lavender / Purple
            "h_range": (125, 155),
            "s_range": (50, 200),
            "v_range": (90, 230),
            "name": "Lavender",
        },
        "200": {  # Bright Yellow-Orange
            "h_range": (18, 32),
            "s_range": (140, 255),
            "v_range": (160, 255),
            "name": "Bright Yellow-Orange",
        },
        "500": {  # Stone Grey (Low saturation, medium brightness)
            "h_range": (0, 180),
            "s_range": (0, 45),
            "v_range": (70, 210),
            "name": "Stone Grey",
        },
    }

    def __init__(self, model_path: str = None, ocr_reader=None):
        """
        Args:
            model_path: Path to custom YOLOv8 currency model weights.
            ocr_reader: Optional existing instance of OCRReader to avoid duplicate loads.
        """
        self.model_path = os.path.abspath(model_path or self.DEFAULT_MODEL_PATH)
        self.yolo_model = None
        self.ocr_reader = ocr_reader

        if YOLO_AVAILABLE and os.path.isfile(self.model_path):
            try:
                print(f"[CurrencyDetector] Loading custom YOLO model from: {self.model_path}")
                self.yolo_model = YOLO(self.model_path)
                print("[CurrencyDetector] Custom YOLO currency model loaded successfully.")
            except Exception as e:
                print(f"[CurrencyDetector] Warning: Failed to load YOLO model: {e}")
                self.yolo_model = None

        if self.yolo_model is None:
            print("[CurrencyDetector] Custom weights 'models/currency_yolov8n.pt' not found.")
            print("                   -> Running multi-modal Fallback Engine (Color Profiling + OCR Numeral Verification).")
            print("                   -> To use custom weights, train YOLOv8 on INR banknotes and place in models/currency_yolov8n.pt")

    def _get_ocr_reader(self):
        """Lazy-load OCR reader if not already injected."""
        if self.ocr_reader is None:
            try:
                from vision.ocr_reader import OCRReader
                self.ocr_reader = OCRReader(gpu=False)
            except ImportError:
                try:
                    from backend.vision.ocr_reader import OCRReader
                    self.ocr_reader = OCRReader(gpu=False)
                except Exception as e:
                    print(f"[CurrencyDetector] Could not load OCRReader for fallback: {e}")
            except Exception as e:
                print(f"[CurrencyDetector] Could not load OCRReader for fallback: {e}")
        return self.ocr_reader

    def _analyze_color_profile(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Analyze central region of frame for dominant banknote color profiles.
        Returns: (best_denomination_candidate, score)
        """
        h, w = frame.shape[:2]
        # Crop central 60% of frame where banknote is typically held
        crop = frame[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = crop.shape[0] * crop.shape[1]

        best_denom = None
        best_ratio = 0.0

        for denom, profile in self.COLOR_PROFILES.items():
            h_min, h_max = profile["h_range"]
            s_min, s_max = profile["s_range"]
            v_min, v_max = profile["v_range"]

            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower, upper)
            match_pixels = cv2.countNonZero(mask)
            ratio = match_pixels / float(total_pixels)

            if ratio > best_ratio:
                best_ratio = ratio
                best_denom = denom

        return best_denom, best_ratio

    def detect_currency(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Detect currency denomination from an OpenCV frame.

        Args:
            frame: A BGR numpy array from OpenCV.

        Returns:
            Tuple of:
              - denomination_str (str): e.g., '100 Rupees', '500 Rupees', or 'No Currency Detected'
              - confidence (float): Recognition confidence between 0.0 and 1.0.
        """
        # ── 1. If custom YOLO model is loaded, run inference ──
        if self.yolo_model is not None:
            results = self.yolo_model(frame, conf=0.50, verbose=False)
            best_conf = 0.0
            best_label = None

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.yolo_model.names.get(cls_id, "").replace("₹", "").strip()
                        if conf > best_conf and label in self.DENOMINATION_NAMES:
                            best_conf = conf
                            best_label = label

            if best_label:
                return self.DENOMINATION_NAMES[best_label], round(best_conf, 2)

        # ── 2. Fallback: Color-Histogram + OCR Numeral Matching ──
        ocr = self._get_ocr_reader()
        extracted_text = ""
        if ocr is not None:
            extracted_text, _ = ocr.read_text(frame)
            extracted_text = extracted_text.upper()

        # Check for explicit numeral patterns in OCR text
        ocr_denom = None
        # Order by largest numbers first to avoid matching '50' inside '500'
        for d in ["500", "200", "100", "50", "20", "10"]:
            # Match word boundary or currency symbol: e.g. "500", "₹500", "RS 500"
            pattern = rf"(?:₹|RS\.?|RUPEES)?\s*\b{d}\b"
            if re.search(pattern, extracted_text):
                ocr_denom = d
                break

        # Check color dominance
        color_denom, color_ratio = self._analyze_color_profile(frame)

        # Decision Fusion
        if ocr_denom is not None:
            # OCR found clear denomination numeral
            if color_denom == ocr_denom and color_ratio > 0.15:
                # Both OCR and Color match!
                return self.DENOMINATION_NAMES[ocr_denom], 0.95
            else:
                # OCR matched clearly
                return self.DENOMINATION_NAMES[ocr_denom], 0.85

        # If OCR didn't catch the exact numeral, check if general currency markers exist
        has_currency_keyword = any(
            kw in extracted_text for kw in ["RESERVE", "BANK", "INDIA", "BHARAT", "RUPEE", "PROMISE"]
        )

        if color_denom and color_ratio > 0.28:
            if has_currency_keyword:
                return f"{self.DENOMINATION_NAMES[color_denom]} (Probable)", round(min(0.80, color_ratio * 2.2), 2)
            elif color_ratio > 0.40:
                return f"{self.DENOMINATION_NAMES[color_denom]} (Color Match)", round(min(0.70, color_ratio * 1.5), 2)

        return "No Currency Detected", 0.0


# ──────────────────────────────────────────────────────────────
#  Self-test entrypoint
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Saarthi — Indian Currency Detector Test")
    print("=" * 55)

    detector = CurrencyDetector()

    # Create synthetic test card with ₹100 Lavender background and text
    h, w = 300, 600
    test_note = np.zeros((h, w, 3), dtype=np.uint8)
    # Lavender in BGR: approximately (180, 140, 160)
    test_note[:] = (180, 140, 160)
    cv2.putText(test_note, "RESERVE BANK OF INDIA", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 30, 40), 2)
    cv2.putText(test_note, "100", (250, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (30, 20, 30), 4)
    cv2.putText(test_note, "ONE HUNDRED RUPEES", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 30, 40), 2)

    print("[Test] Analyzing synthetic banknote...")
    denom, conf = detector.detect_currency(test_note)
    print(f"\n[Test] Result: {denom} (Confidence: {conf})")
    assert "100" in denom, f"Expected 100 Rupees, got {denom}"
    print("[Test] CurrencyDetector self-test complete. ✓")
