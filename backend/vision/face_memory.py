"""
Saarthi — Face Memory Module
==============================
Persistent facial recognition memory using FaceNet embeddings stored
in a local ChromaDB vector database. Allows registering known faces
and recognising them in real-time video frames.

Usage:
    from backend.vision.face_memory import FaceMemory

    memory = FaceMemory()
    memory.register_face("Achal", embedding_vector)
    match = memory.search_face(embedding_vector)
"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

import chromadb
from chromadb.config import Settings
from facenet_pytorch import MTCNN, InceptionResnetV1



# ──────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────

# Persistent storage directory — sits alongside the backend code
_DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(__file__), "..", "chroma_db"
)

# ChromaDB collection name
_COLLECTION_NAME = "faces"


# ──────────────────────────────────────────────────────────────
#  Face Memory — Vector Database Wrapper
# ──────────────────────────────────────────────────────────────

class FaceMemory:
    """
    Face recognition pipeline with persistent vector memory.

    Combines:
        - MTCNN          : face detection & alignment (crops faces from frames)
        - InceptionResnetV1 : 512-dim embedding extraction (pretrained on VGGFace2)
        - ChromaDB       : persistent vector database for face matching

    Each entry stores:
        - id       : unique identifier  (e.g. "face_001")
        - embedding: 512-dim FaceNet vector
        - metadata : {"name": "Achal", ...}

    Args:
        db_dir: Path to the ChromaDB persistence directory.
                Defaults to ``backend/chroma_db/``.
    """

    def __init__(self, db_dir: str = None):
        self.db_dir = os.path.abspath(db_dir or _DEFAULT_DB_DIR)

        # ── Device selection (GPU if available, else CPU) ──
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Face detection & alignment ──
        # MTCNN returns a cropped, aligned 160×160 face tensor
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,         # Return only the most prominent face
            post_process=True,      # Normalise pixel values for FaceNet
            device=self.device,
        )

        # ── Face embedding model ──
        # InceptionResnetV1 pretrained on VGGFace2, frozen for inference
        self.resnet = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)

        print(f"[FaceMemory] Face models loaded on: {self.device}")

        # ── Persistent vector database ──
        os.makedirs(self.db_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.db_dir)

        # Get or create the 'faces' collection
        # Using cosine similarity — standard for FaceNet embeddings
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        face_count = self.collection.count()
        print(f"[FaceMemory] Database loaded from: {self.db_dir}")
        print(f"[FaceMemory] Collection '{_COLLECTION_NAME}' — {face_count} face(s) stored")

    # ──────────────────────────────────────────────────────────
    #  Embedding extraction
    # ──────────────────────────────────────────────────────────

    def get_embedding(self, frame: np.ndarray) -> Optional[list]:
        """
        Extract a 512-dimensional face embedding from an OpenCV frame.

        Pipeline:
            1. Convert BGR (OpenCV) → RGB (PIL Image)
            2. Run MTCNN to detect and crop the most prominent face
            3. Pass the aligned 160×160 crop through InceptionResnetV1
            4. Return the embedding as a plain Python list

        Args:
            frame: A BGR numpy array (H, W, 3) from OpenCV.

        Returns:
            A list of 512 floats (the face embedding), or None if
            no face was detected in the frame.
        """
        # Step 1: BGR → RGB PIL Image
        rgb_frame = frame[:, :, ::-1]  # OpenCV BGR → RGB
        pil_image = Image.fromarray(rgb_frame)

        # Step 2: Detect & crop the face (returns a normalised tensor)
        face_tensor = self.mtcnn(pil_image)

        if face_tensor is None:
            return None  # No face detected

        # Step 3: Add batch dimension and move to device
        face_batch = face_tensor.unsqueeze(0).to(self.device)

        # Step 4: Extract 512-dim embedding (no gradient needed)
        with torch.no_grad():
            embedding = self.resnet(face_batch)

        # Convert to plain Python list for ChromaDB compatibility
        return embedding[0].cpu().numpy().tolist()

    # ──────────────────────────────────────────────────────────
    #  High-level: Enrollment & Identification
    # ──────────────────────────────────────────────────────────

    def enroll_face(self, frame: np.ndarray, name: str) -> Optional[str]:
        """
        Enroll a face from a raw OpenCV frame into the database.

        Extracts the 512-dim embedding and stores it in ChromaDB
        with the person's name as metadata.

        Args:
            frame: A BGR numpy array (H, W, 3) from OpenCV.
            name:  The person's name to associate with this face.

        Returns:
            The face ID string if enrollment succeeded, or None if
            no face was detected in the frame.
        """
        embedding = self.get_embedding(frame)

        if embedding is None:
            print(f"[FaceMemory] Enrollment failed — no face detected in frame.")
            return None

        face_id = self.register_face(name, embedding)
        print(f"[FaceMemory] Enrolled '{name}' successfully.")
        return face_id

    def identify_face(
        self, frame: np.ndarray, threshold: float = 0.6
    ) -> dict:
        """
        Identify a face in a raw OpenCV frame against the database.

        Extracts the embedding, queries ChromaDB for the nearest
        neighbor using cosine distance, and returns the result.

        Args:
            frame:     A BGR numpy array (H, W, 3) from OpenCV.
            threshold: Maximum cosine distance to accept as a match.
                       Lower = stricter. (0.6 is a good default.)

        Returns:
            A dict with:
                - "name"     : Recognized person's name, or "Unknown"
                - "distance" : Cosine distance (lower = more similar)
                - "id"       : Face ID in the database
                - "detected" : True if a face was found in the frame
            If no face is detected in the frame, returns:
                {"name": "No Face", "distance": -1, "id": None, "detected": False}
        """
        embedding = self.get_embedding(frame)

        if embedding is None:
            return {
                "name": "No Face",
                "distance": -1,
                "id": None,
                "detected": False,
            }

        match = self.search_face(embedding, threshold=threshold)

        if match is not None:
            return {
                "name": match["name"],
                "distance": match["distance"],
                "id": match["id"],
                "detected": True,
            }
        else:
            return {
                "name": "Unknown",
                "distance": -1,
                "id": None,
                "detected": True,
            }

    # ──────────────────────────────────────────────────────────
    #  Low-level database operations
    # ──────────────────────────────────────────────────────────

    def register_face(self, name: str, embedding: list, face_id: str = None) -> str:
        """
        Register a new face (or update an existing one) in the database.

        Args:
            name:      Human-readable name for this person.
            embedding: 512-dimensional FaceNet embedding (list of floats).
            face_id:   Optional unique ID. Auto-generated if not provided.

        Returns:
            The ID under which the face was stored.
        """
        if face_id is None:
            count = self.collection.count()
            face_id = f"face_{count + 1:04d}"

        self.collection.upsert(
            ids=[face_id],
            embeddings=[embedding],
            metadatas=[{"name": name}],
        )

        print(f"[FaceMemory] Registered: '{name}' (id={face_id})")
        return face_id

    def search_face(
        self, embedding: list, threshold: float = 0.6
    ) -> Optional[dict]:
        """
        Search for the closest matching face in the database.

        Args:
            embedding: 512-dim query embedding.
            threshold: Maximum cosine distance to accept as a match.
                       Lower = stricter. (0.6 is a good default for FaceNet.)

        Returns:
            A dict {"name": str, "distance": float, "id": str} if a match
            is found within the threshold, otherwise None.
        """
        if self.collection.count() == 0:
            return None

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        best_id = results["ids"][0][0]
        best_dist = results["distances"][0][0]
        best_name = results["metadatas"][0][0].get("name", "Unknown")

        if best_dist > threshold:
            return None

        return {
            "name": best_name,
            "distance": round(best_dist, 4),
            "id": best_id,
        }

    def list_faces(self) -> list:
        """Return a list of all registered faces with their names and IDs."""
        if self.collection.count() == 0:
            return []

        all_data = self.collection.get(include=["metadatas"])
        faces = []
        for fid, meta in zip(all_data["ids"], all_data["metadatas"]):
            faces.append({"id": fid, "name": meta.get("name", "Unknown")})
        return faces

    def delete_face(self, face_id: str):
        """Remove a face from the database by its ID."""
        self.collection.delete(ids=[face_id])
        print(f"[FaceMemory] Deleted face: {face_id}")

    def count(self) -> int:
        """Return the number of faces stored."""
        return self.collection.count()


# ──────────────────────────────────────────────────────────────
#  Quick self-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import sys

    print("=" * 55)
    print("  Saarthi — Face Recognition Pipeline Test")
    print("=" * 55)

    memory = FaceMemory()

    # Grab a single frame from the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Test] Cannot open webcam. Exiting.")
        sys.exit(1)

    print("[Test] Capturing a frame from webcam...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[Test] Failed to capture frame. Exiting.")
        sys.exit(1)

    # Show the captured frame
    cv2.imshow("Captured Frame", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Extract embedding
    print("[Test] Extracting face embedding...")
    embedding = memory.get_embedding(frame)

    if embedding is None:
        print("[Test] No face detected in the frame. Try again with your face visible.")
        sys.exit(0)

    print(f"[Test] Embedding extracted! Length: {len(embedding)}")
    print(f"[Test] First 5 values: {embedding[:5]}")

    # Register the face
    fid = memory.register_face("TestUser", embedding)

    # Search — should find itself
    match = memory.search_face(embedding, threshold=0.6)
    if match:
        print(f"[Test] ✓ Match found: '{match['name']}' (distance={match['distance']})")
    else:
        print("[Test] ✗ No match found (unexpected).")

    # Cleanup
    memory.delete_face(fid)
    print(f"[Test] After cleanup: {memory.count()} face(s)")
    print("\n[Test] Pipeline test complete. ✓")

