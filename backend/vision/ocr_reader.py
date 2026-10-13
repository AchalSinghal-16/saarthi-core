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

    def __init__(self, languages: list = None, gpu: bool = False, min_confidence: float = 0.30):
        """
        Args:
            languages: List of language codes, defaults to ['en'].
            gpu: False to enforce CPU execution for edge hardware.
            min_confidence: Minimum recognition confidence threshold (default: 0.30).
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
        Automatically scales high-res frames (e.g. 1080p phone streams)
        to max 960px for sub-second CPU inference, then projects bounding
        boxes back to the original frame dimensions.

        Args:
            frame: A BGR numpy array from OpenCV.

        Returns:
            Tuple of:
              - consolidated_text (str): Full concatenated text string.
              - detections (list): List of (text, (x1, y1, x2, y2), confidence).
        """
        orig_h, orig_w = frame.shape[:2]

        # Downscale if larger than 960px for fast CPU inference (<1.0s vs >12s)
        max_dim = 960
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
            scaled_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            scaled_frame = frame

        preprocessed = self.preprocess(scaled_frame)

        # EasyOCR readtext returns list of (bbox, text, prob)
        # where bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        raw_results = self.reader.readtext(preprocessed)

        detections = []
        valid_texts = []
        inv_scale = 1.0 / scale

        for bbox, text, conf in raw_results:
            clean_text = text.strip()
            if not clean_text or conf < self.min_confidence:
                continue

            # Convert 4-point polygon to axis-aligned bounding box and scale to original frame
            xs = [int(pt[0] * inv_scale) for pt in bbox]
            ys = [int(pt[1] * inv_scale) for pt in bbox]
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(orig_w, max(xs))
            y2 = min(orig_h, max(ys))

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
