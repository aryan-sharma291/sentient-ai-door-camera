from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from src.utils.timestamp_utils import iso_timestamp
from src.utils.json_utils import to_jsonable

def vertices_to_xyxy(vertices) -> Optional[List[int]]:


    xs = []
    ys = []

    for v in vertices:
        x = getattr(v, "x", None)
        y = getattr(v, "y", None)

        if x is not None:
            xs.append(int(x))
        if y is not None:
            ys.append(int(y))

    # Guard: if we cannot form a box, return None
    if not xs or not ys:
        return None

    return [min(xs), min(ys), max(xs), max(ys)]

likelihood_score = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 1,
    "UNLIKELY": 2,
    "POSSIBLE": 3,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}

def normalize_google_faces(face_annotations):
    normalized_faces = []

    for face in face_annotations:
        bbox = vertices_to_xyxy(face.bounding_poly.vertices)
        if bbox is None:
            continue
        normalized_faces.append({
            "source": "Google Vision",
            "bbox_xyxy": bbox,
            "confidence": float(getattr(face, "detection_confidence", 0.0)),
            "emotion": {
                "joy": str(face.joy_likelihood),
                "anger": str(face.anger_likelihood),
                "sorrow": str(face.sorrow_likelihood),
                "surprise": str(face.surprise_likelihood),
            },
            "quality": {
                "blurred": str(face.blurred_likelihood),
                "underexposed": str(face.under_exposed_likelihood),
            }
        })
    return normalized_faces

def build_verdict(faces, objects):
    person = any(o["label"].lower() == "person" and o["confidence"] >= 0.5 for o in objects)

    face = len(faces) > 0
    level = "HIGH" if person else "LOW"
    return {"person_detected": person, "face_detected": face, "level": level}

def build_event_record(raw_path, processed_path, img_wh, faces, objects):
    return {
        "timestamp": iso_timestamp() ,
        "image": {"raw_path": raw_path, "processed_path": processed_path, "width": img_wh[0], "height": img_wh[1]},
        "faces": faces,
         "objects": objects,
        "verdict": build_verdict(faces, objects)
    }

def score_frame(faces: List[Dict], image_wh: Tuple[int, int]) -> float:
    w, h = image_wh
    if not faces:
        return 0.0

    best = 0.0
    for f in faces:
        x1, y1, x2, y2 = f["bbox_xyxy"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = abs(cx - (w / 2.0)) / max(1.0, w)
        dy = abs(cy - (h / 2.0)) / max(1.0, h)
        center_bonus = 1.0 - (dx + dy) / 2.0  # 0..1-ish

        # main score = face area + centered bonus
        s = area * (0.5 + center_bonus)
        if s > best:
            best = s

    return float(best)