from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import face_recognition

from src.utils.json_utils import read_json, safe_write_json
KNOWN_FACES_DIR = Path("data/known_faces")
ENCODING_PATH = KNOWN_FACES_DIR / "encodings.json"
DEFAULT_TOLERANCE = 0.525

@dataclass
class FaceMatch:
    name: str
    confidence: float
    bbox_xyxy: List[int]

def iter_known_faces(known_dir: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    if not known_dir.exists():
        return items
    for person_dir in known_dir.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((name, img_path))
    return items

def build_encodings(known_dir: Path = KNOWN_FACES_DIR) -> Dict[str, Any]:
    entries = []
    for name, img_path in iter_known_faces(known_dir):
        image = face_recognition.load_image_file(str(img_path))
        encs = face_recognition.face_encodings(image)
        if not encs:
            continue
        entries.append({
            "name": name,
            "path": str(img_path),
            "encoding": encs[0].tolist(),
        })

    payload = {"version": 1, "known_dir": str(known_dir), "entries": entries}
    safe_write_json(ENCODING_PATH, payload, pretty=True)
    return payload

def load_encodings() -> Dict[str, Any]:
    cached = read_json(ENCODING_PATH, default=None)
    if cached and cached.get("entries"):
        return cached
    return build_encodings(KNOWN_FACES_DIR)

def prepare_known_arrays(cache: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    vectors: List[np.ndarray] = []
    for entry in cache.get("entries", []):
        names.append(entry["name"])
        vectors.append(np.array(entry["encoding"], dtype=np.float32))
    if not vectors:
        return [], np.zeros((0, 128), dtype=np.float32)
    return names, np.vstack(vectors)

def recognize_faces_offline(image_rgb: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> List[FaceMatch]:
    cache = load_encodings()
    names, vectors = prepare_known_arrays(cache)

    face_locations = face_recognition.face_locations(image_rgb, model="cnn")
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    results: List[FaceMatch] = []
    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        bbox = [int(left), int(top), int(right), int(bottom)]

        if len(names) == 0:
            results.append(FaceMatch(name="UNKNOWN", confidence=0.0, bbox_xyxy=bbox))
            continue

        distances = face_recognition.face_distance(vectors, enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        confidence = max(0.0, 1.0 - (best_dist / tolerance))
        if best_dist <= tolerance:
            results.append(FaceMatch(name=names[best_idx], confidence=confidence, bbox_xyxy=bbox))
        else:
            results.append(FaceMatch(name="UNKNOWN", confidence=confidence, bbox_xyxy=bbox))

    return results
