import cv2
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
from pathlib import Path

def preprocess_image(raw_path: Path, processed_dir: Path, time_stamp) -> tuple[Path, dict]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image not found: {raw_path}")
    image = Image.open(raw_path)
    image = ImageOps.exif_transpose(image)
    good_image = np.array(image)
    opencv_image = cv2.cvtColor(good_image, cv2.COLOR_RGB2BGR)
    target_width = 640
    target_height = 480
    opencv_image = cv2.resize(opencv_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 25:
        opencv_image = cv2.convertScaleAbs(opencv_image, alpha=1.2, beta=25)
    opencv_image = cv2.fastNlMeansDenoisingColored(opencv_image, None, 10, 10, 7, 21)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_file = processed_dir / f"{raw_path.stem}__processed__{time_stamp}.jpg"
    write = cv2.imwrite(str(processed_file), opencv_image)
    if not write:
        raise RuntimeError(f"Failed to write processed image: {processed_file}")
    h, w = opencv_image.shape[:2]
    meta = {
    "raw_path": str(raw_path),
    "processed_path": str(processed_file),
    "width": w,
    "height": h,
    "brightness_before": brightness,
    "timestamp": time_stamp,
    }
    return processed_file, meta
