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

    DENOMINATION_PATTERNS = {
        "500": [
            r"\b500\b", r"₹\s*500", r"\b5OO\b", r"\bSOO\b", r"\bFIVE\s*HUNDRED\b",
            r"\b500\s*RUPEES?\b", r"५००"
        ],
        "200": [
            r"\b200\b", r"₹\s*200", r"\b2OO\b", r"\bZOO\b", r"\bTWO\s*HUNDRED\b",
            r"\b200\s*RUPEES?\b", r"२००"
        ],
        "100": [
            r"\b100\b", r"₹\s*100", r"\b1OO\b", r"\bIOO\b", r"\bONE\s*HUNDRED\b",
            r"\bHUNDRED\b", r"\b100\s*RUPEES?\b", r"१००"
        ],
        "50": [
            r"\b50\b", r"₹\s*50", r"\b5O\b", r"\bSO\b", r"\bFIFTY\b",
            r"\b50\s*RUPEES?\b", r"५०"
        ],
        "20": [
            r"\b20\b", r"₹\s*20", r"\b2O\b", r"\bZO\b", r"\bTWENTY\b",
            r"\b20\s*RUPEES?\b", r"२०"
        ],
        "10": [
            r"\b10\b", r"₹\s*10", r"\b1O\b", r"\bIO\b", r"\bTEN\b",
            r"\b10\s*RUPEES?\b", r"१०"
        ],
    }

    BANKNOTE_KEYWORDS = [
        "RESERVE", "BANK", "INDIA", "BHARAT", "RUPEE", "RUPEES",
        "PROMISE", "CENTRAL", "GOVERNOR", "MAHATMA", "GANDHI",
        "GUARANTEED", "PAY", "BEARER", "SUM"
    ]

    # HSV Color profiles for Mahatma Gandhi New Series banknotes
    # Tuned specifically to avoid false positives on ambient neutral backgrounds
    COLOR_PROFILES = {
        "20": {   # Greenish Yellow (distinct yellow-green hue)
            "h_range": (26, 52),
            "s_range": (55, 240),
            "v_range": (75, 240),
            "name": "Greenish Yellow",
        },
        "50": {   # Fluorescent Blue / Cyan
            "h_range": (85, 115),
            "s_range": (70, 255),
            "v_range": (80, 255),
            "name": "Fluorescent Blue",
        },
        "100": {  # Lavender / Purple
            "h_range": (125, 155),
            "s_range": (45, 200),
            "v_range": (75, 230),
            "name": "Lavender",
        },
        "200": {  # Bright Yellow-Orange
            "h_range": (15, 26),
            "s_range": (120, 255),
            "v_range": (140, 255),
            "name": "Bright Yellow-Orange",
        },
        "10": {   # Chocolate Brown
            "h_range": (8, 22),
            "s_range": (60, 220),
            "v_range": (50, 180),
            "name": "Chocolate Brown",
        },
        "500": {  # Stone Grey (Requires strict saturation bounds)
            "h_range": (50, 110),
            "s_range": (15, 50),
            "v_range": (80, 180),
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
              - denomination_str (str): e.g., '20 Rupees', '100 Rupees', or 'No Currency Detected'
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

        # ── 2. Fallback: OCR Numeral/Text Matching + Color Profiling ──
        ocr = self._get_ocr_reader()
        extracted_text = ""
        if ocr is not None:
            extracted_text, _ = ocr.read_text(frame)
            extracted_text = extracted_text.upper()

        # Step 2A: Match against denomination text/numeral patterns
        ocr_denom = None
        for denom, patterns in self.DENOMINATION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, extracted_text):
                    ocr_denom = denom
                    break
            if ocr_denom is not None:
                break

        # Step 2B: Check color dominance in central zone
        color_denom, color_ratio = self._analyze_color_profile(frame)

        # Step 2C: Check for official banknote keywords
        has_banknote_keyword = any(
            kw in extracted_text for kw in self.BANKNOTE_KEYWORDS
        )

        # ── Decision Fusion ──
        # Case 1: Direct OCR Match
        if ocr_denom is not None:
            if color_denom == ocr_denom and color_ratio > 0.12:
                # Both OCR and Color match!
                return self.DENOMINATION_NAMES[ocr_denom], 0.95
            else:
                # Strong OCR match
                return self.DENOMINATION_NAMES[ocr_denom], 0.88

        # Case 2: Banknote keywords detected (e.g. "RESERVE BANK OF INDIA") + Color dominance
        if has_banknote_keyword and color_denom is not None:
            # Color provides the denomination
            if color_ratio > 0.15:
                return f"{self.DENOMINATION_NAMES[color_denom]} (Probable)", round(min(0.85, 0.65 + color_ratio), 2)

        # Case 3: High-confidence non-grey saturated color match (e.g. ₹20 Greenish-Yellow or ₹50 Cyan)
        # Note: Exclude "500" from color-only match to prevent false positives from neutral room backgrounds
        if color_denom is not None and color_denom != "500" and color_ratio > 0.35:
            return f"{self.DENOMINATION_NAMES[color_denom]} (Color Match)", round(min(0.75, color_ratio * 1.5), 2)

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
