# main.py
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2

from src.camera.capture_still import capture_burst
from src.sensors.pir_sensor import PIRSensor
from src.sensors.led_control import LEDControl

from src.cloud.google_vision_client import GoogleVisionClient
from src.ai.postprocess import normalize_google_faces, build_event_record, score_frame
from src.ai.offline_face_recognition import recognize_faces_offline

from src.notifications.telegram_notifier import send_event_alert, flush_outbox


# -----------------------------
# Paths / Settings
# -----------------------------
RAW_DIR = Path("data/images/raw")
PROCESSED_DIR = Path("data/images/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BURST_COUNT = 6
BURST_INTERVAL_S = 0.15
MOTION_COOLDOWN_S = 2.0

# If you want, keep this to filter objects later
PERSON_CONFIDENCE_MIN = 0.50


# -----------------------------
# Helpers for saving "processed" images
# -----------------------------
def _safe_text(x: Any) -> str:
    return "" if x is None else str(x)


def _get_bbox_xyxy(face_or_obj: Dict[str, Any]) -> Optional[List[int]]:
    """
    Accepts your normalized faces dicts (bbox_xyxy)
    and *optionally* object dicts if you later add bbox.
    """
    bb = face_or_obj.get("bbox_xyxy")
    if isinstance(bb, list) and len(bb) == 4:
        return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    return None


def _draw_box(img_bgr, bbox_xyxy: List[int], label: str) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        # Put label above box if possible
        y_text = max(15, y1 - 8)
        cv2.putText(
            img_bgr,
            label[:60],
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def save_processed_image(
    raw_path: str,
    processed_dir: Path,
    faces: List[Dict[str, Any]],
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reads raw image, draws face boxes + labels, saves it into data/images/processed,
    returns processed_path + width/height.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(raw_path)
    if img_bgr is None:
        # If read fails, just return empty processed
        return {"processed_path": "", "width": 0, "height": 0}

    h, w = img_bgr.shape[0], img_bgr.shape[1]

    # Draw faces (your normalized faces from postprocess.normalize_google_faces)
    for f in faces:
        bb = _get_bbox_xyxy(f)
        if not bb:
            continue

        # label: name (offline) + confidence + top emotion if present
        name = _safe_text(f.get("name"))
        conf = f.get("confidence", 0.0)
        emo = f.get("emotion") or {}
        # Google Vision gives likelihood strings; pick a few to show
        emo_bits = []
        for k in ["joy", "anger", "sorrow", "surprise"]:
            if k in emo and _safe_text(emo[k]):
                emo_bits.append(f"{k}:{_safe_text(emo[k])}")
        emo_text = " ".join(emo_bits[:2])  # keep it short

        label_parts = []
        if name:
            label_parts.append(name)
        label_parts.append(f"{conf:.2f}")
        if emo_text:
            label_parts.append(emo_text)
        label = " | ".join(label_parts)

        _draw_box(img_bgr, bb, label)

    # Objects:
    # Right now your Google Vision objects list likely contains label/confidence only.
    # If later you add bbox for objects, this will automatically draw them too.
    for o in objects:
        bb = _get_bbox_xyxy(o)
        if not bb:
            continue
        label = _safe_text(o.get("label")) or _safe_text(o.get("name"))
        conf = o.get("confidence", 0.0)
        _draw_box(img_bgr, bb, f"{label} {conf:.2f}")

    # Write processed image
    raw_name = Path(raw_path).name
    stem = Path(raw_name).stem
    processed_path = processed_dir / f"{stem}_processed.jpg"
    cv2.imwrite(str(processed_path), img_bgr)

    return {"processed_path": str(processed_path), "width": int(w), "height": int(h)}


# -----------------------------
# Choosing best frame
# -----------------------------
def choose_best_by_face_score(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_score = -1.0

    for c in candidates:
        faces = c.get("faces", [])
        w = int(c.get("width") or 0)
        h = int(c.get("height") or 0)

        # If width/height missing, try reading image quickly
        if (w == 0 or h == 0) and c.get("raw_path"):
            img = cv2.imread(c["raw_path"])
            if img is not None:
                h = int(img.shape[0])
                w = int(img.shape[1])

        s = score_frame(faces, (w, h))

        if s > best_score:
            best_score = s
            best = c

    return best or candidates[0]


# -----------------------------
# Google Vision on burst
# -----------------------------
def run_google_on_burst(gv_client: GoogleVisionClient, burst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for shot in burst:
        raw_path = shot["raw_path"]

        gv = gv_client.analyze_image_path(raw_path)

        face_annotations = gv.get("face_annotations", [])
        objects = gv.get("objects", [])

        faces = normalize_google_faces(face_annotations)

        width = gv.get("width")
        height = gv.get("height")

        results.append(
            {
                "raw_path": raw_path,
                "faces": faces,
                "objects": objects,
                "width": width,
                "height": height,
                "google_ok": True,
            }
        )

    return results


# -----------------------------
# Offline fallback (only when Google fails)
# -----------------------------
def offline_fallback_for_burst(burst: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Use the middle frame of burst
    mid = burst[len(burst) // 2]
    raw_path = mid["raw_path"]

    img_bgr = cv2.imread(raw_path)
    if img_bgr is None:
        return {
            "raw_path": raw_path,
            "faces": [],
            "objects": [],
            "width": 0,
            "height": 0,
            "google_ok": False,
            "wifi_failed": True,
        }

    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    matches = recognize_faces_offline(image_rgb)

    faces: List[Dict[str, Any]] = []
    for m in matches:
        faces.append(
            {
                "source": "offline_face_recognition",
                "bbox_xyxy": m.bbox_xyxy,
                "confidence": float(m.confidence),
                "name": m.name,
                "emotion": {},  # offline face_recognition does NOT provide emotion
                "quality": {},
            }
        )

    h, w = image_rgb.shape[0], image_rgb.shape[1]

    return {
        "raw_path": raw_path,
        "faces": faces,
        "objects": [],
        "width": int(w),
        "height": int(h),
        "google_ok": False,
        "wifi_failed": True,
    }


# -----------------------------
# Main loop
# -----------------------------
def run() -> None:
    pir = PIRSensor(pin=17, warmup_seconds=2.0)
    led = LEDControl(pin=27)
    gv_client = GoogleVisionClient()

    pir.warm_up()

    while True:
        # 1) Wait for motion
        pir.wait_for_motion()

        # 2) Turn light on
        led.on()

        # 3) Capture burst
        burst = capture_burst(
            RAW_DIR,
            prefix="burst",
            burst_count=BURST_COUNT,
            interval_s=BURST_INTERVAL_S,
        )

        # 4) Try Google Vision across burst
        used_fallback = False
        try:
            google_results = run_google_on_burst(gv_client, burst)
            best = choose_best_by_face_score(google_results)
        except Exception:
            used_fallback = True
            best = offline_fallback_for_burst(burst)

        # 5) Save processed image (boxes/labels) and fill processed_path
        processed_info = save_processed_image(
            raw_path=best["raw_path"],
            processed_dir=PROCESSED_DIR,
            faces=best.get("faces", []),
            objects=best.get("objects", []),
        )

        processed_path = processed_info["processed_path"]
        width = best.get("width") or processed_info["width"]
        height = best.get("height") or processed_info["height"]

        # 6) Build event record (THIS is where processed_path gets populated)
        img_wh = (int(width or 0), int(height or 0))
        event = build_event_record(
            raw_path=best["raw_path"],
            processed_path=processed_path,
            img_wh=img_wh,
            faces=best.get("faces", []),
            objects=best.get("objects", []),
        )

        # Add wifi status note (so the user knows fallback happened)
        if used_fallback:
            event["wifi_status"] = "WIFI_DOWN_USED_OFFLINE_FALLBACK"
        else:
            event["wifi_status"] = "WIFI_OK_USED_GOOGLE_VISION"

        # 7) Telegram alert (send raw image; you can switch to processed_path if you prefer)
        send_event_alert(event, raw_image_path=processed_path or best["raw_path"])

        # 8) Turn light off
        led.off()

        # 9) Try to flush queued messages (if Wi-Fi came back)
        try:
            flush_outbox(max_send=20)
        except Exception:
            pass

        # 10) Cooldown so you don’t spam captures
        # Wait until motion stops, then sleep a bit
        pir.wait_for_no_motion(timeout=MOTION_COOLDOWN_S)
        time.sleep(MOTION_COOLDOWN_S)


if __name__ == "__main__":
    run()

