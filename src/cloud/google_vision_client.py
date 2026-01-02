# export GOOGLE_APPLICATION_CREDENTIALS="/Users/aryansharma/MySecondProject/SentientAI/Secret/sentientai-481122-3d9a6f8d91a9.json"

from __future__ import annotations
from src.utils.env_loader import load_api_keys
load_api_keys()
from dataclasses import dataclass

from pathlib import Path

from typing import List, Optional, Dict, Any, Tuple

import os

from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import vision

@dataclass
class VisionConfig:
    credentials_path: Optional[str] = None
    timeout_seconds: int = 15
class GoogleVisionClient:
    def __init__(self,  config: Optional[VisionConfig] = None):
        load_dotenv()

        self.config = config or VisionConfig()
        creds = self.config.credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not creds:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set in .env")
        if not os.path.exists(creds):
            raise RuntimeError(f"Credential file not found at: {creds}")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
        creds = service_account.Credentials.from_service_account_file(creds)
        self.client = vision.ImageAnnotatorClient(credentials=creds)

    def detect_faces(self, image_bytes: bytes):
        image = vision.Image(content=image_bytes)
        resp = self.client.face_detection(image=image)
        if resp.error.message:
            raise RuntimeError(f"Vision face_detection error: {resp.error.message}")
        return resp.face_annotations

    def detect_labels(self, image_bytes: bytes, max_results: int = 10):
        image = vision.Image(content=image_bytes)
        resp = self.client.label_detection(image=image, max_results=max_results)

        if resp.error.message:
            raise RuntimeError(f"Vision label_detection error: {resp.error.message}")

        labels = []
        for label in resp.label_annotations:
            labels.append({
                "label": label.description,
                "confidence": float(label.score),
            })
        return labels
    def detect_objects(self, image_bytes: bytes):
        image = vision.Image(content=image_bytes)
        resp = self.client.object_localization(image=image)
        if resp.error.message:
            raise RuntimeError(f"Vision object_localization error: {resp.error.message}")
        objects = []
        for obj in resp.label_annotations:
            objects.append({
                "label": obj.description,
                "confidence": float(obj.score),
            })
        return objects
    def analyze_image_path(self, image_path: str | Path) -> Dict[str, Any]:
        image_bytes = _read_image_bytes(image_path)

        faces = self.detect_faces(image_bytes)
        labels = self.detect_labels(image_bytes, max_results=10)
        objects = self.detect_objects(image_bytes)

        return {
            "faces": faces,
            "labels": labels,
            "objects": objects,
        }



def _read_image_bytes(path: str | Path) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_bytes()

client2 = GoogleVisionClient()
result = client2.analyze_image_path("/Users/aryansharma/MySecondProject/SentientAI/src/cloud/picture1.jpeg")
print(result["labels"][:5])
print(result["objects"][:5])
print(len(result["faces"]))

