"""
Saarthi — Threaded Video Stream Reader
=======================================
A multi-threaded video capture class that reads frames from a network
stream (e.g., IP Webcam app) or a local webcam on a dedicated background
thread. Only the most recent frame is kept, discarding stale buffered
frames to eliminate OpenCV's internal buffer lag.

Usage:
    python backend/vision/stream_reader.py --source "http://<PHONE_IP>:8080/video"
    python backend/vision/stream_reader.py --source 0
"""

import sys
import time
import argparse
import threading

import cv2
import numpy as np


class ThreadedStreamReader:
    """
    Continuously captures frames from a video source on a background
    thread and exposes only the latest frame to the caller.

    Attributes:
        source:  Stream URL string or integer device index.
        frame:   The most recently captured frame (numpy array or None).
        running: Whether the capture thread is actively reading.
    """

    def __init__(self, source=0):
        """
        Args:
            source: A network stream URL (str) or local camera index (int).
                    Examples:
                        - "http://192.168.1.5:8080/video"  (IP Webcam)
                        - 0  (default laptop webcam)
        """
        self.source = source
        self.frame = None
        self.running = False

        # Lock to ensure thread-safe reads/writes of self.frame
        self._lock = threading.Lock()
        self._thread = None
        self._cap = None

    def start(self):
        """Open the video source and begin capturing on a background thread."""
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise ConnectionError(
                f"[StreamReader] ERROR: Cannot open video source: {self.source}"
            )

        # Minimise OpenCV's internal buffer to 1 frame so we always
        # grab the freshest frame available from the stream.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

        print(f"[StreamReader] Started capture from: {self.source}")
        return self

    def _update(self):
        """
        Background worker: continuously grabs frames and overwrites
        the shared `self.frame` variable so only the newest frame
        is ever available to consumers.
        """
        while self.running:
            ret, frame = self._cap.read()
            if not ret:
                print("[StreamReader] WARNING: Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            with self._lock:
                self.frame = frame

        # Cleanup when the loop exits
        self._cap.release()

    def read(self):
        """
        Return the most recent frame in a thread-safe manner.

        Returns:
            frame (numpy.ndarray | None): The latest BGR frame, or None
                                          if no frame has been captured yet.
        """
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Signal the background thread to stop and wait for it to finish."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        print("[StreamReader] Capture stopped.")


# ──────────────────────────────────────────────────────────────
#  Entry point — quick test / demo
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saarthi Threaded Video Stream Reader"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Video source: a network stream URL "
            "(e.g. http://<PHONE_IP>:8080/video) "
            "or an integer camera index (default: 0 for local webcam)."
        ),
    )
    args = parser.parse_args()

    # If the source is a plain integer string, cast it to int for OpenCV
    source = int(args.source) if args.source.isdigit() else args.source

    reader = ThreadedStreamReader(source=source)

    try:
        reader.start()
        print("[Main] Press 'q' in the display window to quit.\n")

        while True:
            frame = reader.read()

            if frame is None:
                # Stream hasn't delivered its first frame yet
                time.sleep(0.01)
                continue

            # Overlay a small status label
            cv2.putText(
                frame,
                "Saarthi Stream [LIVE]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Saarthi — Stream Reader", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except ConnectionError as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("[Main] Clean shutdown complete.")
