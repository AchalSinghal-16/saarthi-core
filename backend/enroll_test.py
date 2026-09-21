"""
Saarthi — Face Enrollment Utility
====================================
Connects to a video stream (IP Webcam or local webcam), displays a
live preview, and lets the user enroll their face into the ChromaDB
vector database by typing their name and pressing Enter.

Usage:
    python backend/enroll_test.py --source "http://192.168.1.7:8080/video"
    python backend/enroll_test.py --source 0
"""

import sys
import time
import argparse

import cv2

from vision.stream_reader import ThreadedStreamReader
from vision.face_memory import FaceMemory


def main():
    parser = argparse.ArgumentParser(
        description="Saarthi — Enroll a face into the recognition database"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: stream URL or camera index (default: 0).",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 55)
    print("   Saarthi — Face Enrollment")
    print("=" * 55)

    # Initialize components
    memory = FaceMemory()
    reader = ThreadedStreamReader(source=source)

    try:
        reader.start()
        print()

        while True:
            # ── Prompt for name ──
            print("-" * 40)
            name = input("  Enter name to enroll (or 'q' to quit): ").strip()

            if name.lower() == "q":
                print("\n  [Enroll] Exiting.")
                break

            if not name:
                print("  [Enroll] Name cannot be empty. Try again.")
                continue

            # ── Countdown with live preview ──
            print(f"  [Enroll] Look at the camera! Capturing in 3 seconds...")

            captured_frame = None
            start = time.time()

            while True:
                frame = reader.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                elapsed = time.time() - start
                remaining = max(0, 3.0 - elapsed)

                # Draw countdown on preview
                display = frame.copy()
                cv2.putText(
                    display,
                    f"Enrolling: {name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Capturing in {remaining:.1f}s ...",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Saarthi — Enrollment Preview", display)
                cv2.waitKey(1)

                if elapsed >= 3.0:
                    captured_frame = frame.copy()
                    break

            # ── Show captured frame ──
            cv2.putText(
                captured_frame,
                "CAPTURED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            cv2.imshow("Saarthi — Enrollment Preview", captured_frame)
            cv2.waitKey(500)

            # ── Enroll ──
            print(f"  [Enroll] Processing face for '{name}'...")
            face_id = memory.enroll_face(captured_frame, name)

            if face_id:
                print(f"  [Enroll] ✓ '{name}' enrolled successfully! (ID: {face_id})")
            else:
                print(f"  [Enroll] ✗ No face detected. Please try again.")

            # Show all enrolled faces
            faces = memory.list_faces()
            print(f"\n  [Database] Total enrolled faces: {len(faces)}")
            for f in faces:
                print(f"    - {f['id']}: {f['name']}")
            print()

    except ConnectionError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  [Enroll] Interrupted.")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("  [Enroll] Done.\n")


if __name__ == "__main__":
    main()
