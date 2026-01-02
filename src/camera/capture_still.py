

from picamera2 import Picamera2
from pathlib import Path
from src.utils.timestamp_utils import iso_timestamp
import time
from typing import Optional, Dict, Any, Tuple, List
import os


def capture_still(
    raw_dir: Path,
    prefix: str = "capture",
    return_array: bool = False,
    warmup_s: float = 0.1,
) -> Dict[str, Any]:

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ts = iso_timestamp().replace(":", "-")
    filename = f"{prefix}_{ts}.jpg"
    raw_path = raw_dir / filename

    picam = Picamera2()
    try:
        config = picam.create_still_configuration()
        picam.configure(config)

        picam.start()
        time.sleep(max(0.0, warmup_s))

        # Save JPEG to disk
        picam.capture_file(str(raw_path))

        img = None
        width = None
        height = None

        if return_array:
            # Capture a frame as a NumPy array (H x W x C)
            img = picam.capture_array()
            if img is not None and hasattr(img, "shape") and len(img.shape) >= 2:
                height = int(img.shape[0])
                width = int(img.shape[1])

        return {
            "timestamp": ts,
            "raw_path": str(raw_path),
            "width": width,
            "height": height,
            "array": img,
        }

    finally:
        # Always clean up camera resources
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass

def capture_burst(raw_dir: Path,
    prefix: str = "burst",
    burst_count: int = 6,
    interval_s: float = 0.15,
) -> List[Dict[str, Any]]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    time.sleep(0.1)

    results: List[Dict[str, Any]] = []
    base_ts = iso_timestamp().replace(":", "-")
    for i in range(burst_count):
        ts = iso_timestamp().replace(":", "-")
        filename = f"{prefix}_{base_ts}_{i:02d}_{ts}.jpg"
        raw_path = raw_dir / filename
        cam.capture_file(str(raw_path))
        results.append({"timestamp": ts, "raw_path": str(raw_path)})
        time.sleep(interval_s)

    cam.stop()
    cam.close()
    return results