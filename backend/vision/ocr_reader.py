"""
Saarthi — Smart OCR Reader Module
==================================
Extracts localized text blocks from camera frames using EasyOCR.
Optimized for reading documents, medication packaging, product labels,
and street signage for visually impaired individuals.

Runs on CPU (gpu=False) for lightweight deployment on wearable edge devices.
"""

from typing import List, Tuple
import cv2
import numpy as np
import easyocr


class OCRReader:
    """
    On-demand optical character recognition reader.

    Preprocesses image frames with grayscale conversion and contrast
    enhancement (CLAHE), runs EasyOCR text extraction, and filters out
    low-confidence artifacts (< 0.40).
    """

    def __init__(self, languages: list = None, gpu: bool = False, min_confidence: float = 0.40):
        """
        Args:
            languages: List of language codes, defaults to ['en'].
            gpu: False to enforce CPU execution for edge hardware.
            min_confidence: Minimum recognition confidence threshold (default: 0.40).
        """
        self.languages = languages or ['en']
        self.gpu = gpu
        self.min_confidence = min_confidence

        print(f"[OCRReader] Initializing EasyOCR reader on CPU (languages={self.languages})...")
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        print("[OCRReader] EasyOCR initialized successfully.")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess input frame for enhanced character edge contrast.
        - Converts BGR to Grayscale
        - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # CLAHE for localized illumination balancing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def read_text(self, frame: np.ndarray) -> Tuple[str, List[Tuple[str, Tuple[int, int, int, int], float]]]:
        """
        Extract text from an OpenCV BGR frame.

        Args:
            frame: A BGR numpy array from OpenCV.

        Returns:
            Tuple of:
              - consolidated_text (str): Full concatenated text string.
              - detections (list): List of (text, (x1, y1, x2, y2), confidence).
        """
        preprocessed = self.preprocess(frame)

        # EasyOCR readtext returns list of (bbox, text, prob)
        # where bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        raw_results = self.reader.readtext(preprocessed)

        detections = []
        valid_texts = []

        for bbox, text, conf in raw_results:
            clean_text = text.strip()
            if not clean_text or conf < self.min_confidence:
                continue

            # Convert 4-point polygon to axis-aligned bounding box (x1, y1, x2, y2)
            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(w, max(xs))
            y2 = min(h, max(ys))

            detections.append((clean_text, (x1, y1, x2, y2), round(float(conf), 3)))
            valid_texts.append(clean_text)

        consolidated_text = " ".join(valid_texts)
        return consolidated_text, detections


# ──────────────────────────────────────────────────────────────
#  Self-test entrypoint
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Saarthi — Smart OCR Reader Test")
    print("=" * 55)

    reader = OCRReader()

    # Create synthetic test image with clean text
    test_img = np.zeros((200, 600, 3), dtype=np.uint8)
    test_img[:] = (245, 245, 245)
    cv2.putText(test_img, "PARACETAMOL 500MG", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(test_img, "SAARTHI EDGE AI", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 120, 0), 2)

    print("[Test] Reading test image text...")
    full_text, boxes = reader.read_text(test_img)

    print(f"\n[Test] Extracted Text: '{full_text}'")
    print(f"[Test] Detections count: {len(boxes)}")
    for txt, box, conf in boxes:
        print(f"  - '{txt}' at {box} (confidence: {conf})")

    print("\n[Test] OCRReader self-test complete. ✓")
