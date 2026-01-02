
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
OUTBOX_PATH = Path("notifications/queue/telegram_outbox.jsonl")

from typing import Any, Dict, Optional, List
import time
import json
from telegram import Bot, Update
from telegram.error import NetworkError, TimedOut, RetryAfter, TelegramError

from src.utils.timestamp_utils import iso_timestamp
from src.utils.json_utils import append_jsonl, read_json, safe_write_json
from src.utils.env_loader import load_api_keys
@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
def load_telegram_config() -> TelegramConfig:
    keys = load_api_keys()

    bot_token = keys.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = keys.get("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

    return TelegramConfig(bot_token, chat_id)
LIKELIHOOD_SCORE = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 1,
    "UNLIKELY": 2,
    "POSSIBLE": 3,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}

def _emotion_value(v:Any) -> int:
    if v is None:
        return 0
    s = str(v).strip().upper()
    return LIKELIHOOD_SCORE.get(s, 0)
def summarize_emotion(faces: List[Dict[str, Any]]) -> Dict[str, Any]:

    totals: Dict[str, int] = {}
    for face in faces:
        emo = (face.get("emotion") or {})
        for name, val in emo.items():
            totals[name] = totals.get(name, 0) + _emotion_value(val)

    if not totals:
        return {"top_emotions": [], "face_count": len(faces)}

    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    top3 = ranked[:3]
    return {"top_emotions": top3, "face_count": len(faces)}


def build_alert_text(event: Dict[str, Any]) -> str:
    verdict = event.get("verdict") or {}
    faces = event.get("faces") or []
    objects = event.get("objects") or []

    emo_summary = summarize_emotion(faces)
    top = emo_summary["top_emotions"]

    if top:
        emo_line = ", ".join(f"{name}:{score}" for name, score in top)
    else:
        emo_line = "None"
    top_labels = []
    for o in objects[:5]:
        label = str(o.get("label") or o.get("name") or o.get("description") or "").strip()
        confidence = o.get("confidence") or o.get("value") or 0
        if label:
            top_labels.append(f"{label} ({confidence:.2f})")
        else:
            top_labels.append(label)
    objects_line = ", ".join(top_labels) if top_labels else "None"
    return (
        f"SentientAI Alert\n"
        f"Time: {iso_timestamp()}\n"
        f"Risk Level: {verdict.get('level', 'UNKNOWN')}\n"
        f"Person Detected: {verdict.get('person_detected', False)}\n"
        f"Face Detected: {verdict.get('face_detected', False)}\n"
        f"Top Emotions: {emo_line}\n"
        f"Top Objects: {objects_line}"
    )
def enqueue_alert(job: Dict[str, Any]) -> None:
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(OUTBOX_PATH, job)
def send_now(bot: Bot, chat_id: str, text: str, photo_path: Optional[str] = None) -> None:
    if photo_path:
        p = Path(photo_path)
        if p.exists():
            with open(photo_path, "rb") as photo:
                bot.send_photo(chat_id=chat_id, photo=photo, caption=text)
            return
    bot.send_message(chat_id=chat_id, text=text)

def send_event_alert(event: Dict[str, Any], raw_image_path: Optional[str] = None) -> None:
    cfg = load_telegram_config()
    bot = Bot(token=cfg.bot_token)
    text = build_alert_text(event)

    try:
        send_now(bot, cfg.chat_id, text, photo_path=raw_image_path)

    except RetryAfter as e:
        wait_s = int(getattr(e, "retry_after", 30) or 30)
        job = {
            "created_at": iso_timestamp(),
            "attempts": 0,
            "next_try_at": iso_timestamp(),
            "text": text,
            "photo_path": raw_image_path,
            "reason": f"RetryAfter: {getattr(e, 'retry_after', None)}",
        }
        enqueue_alert(job)
    except (NetworkError, TimedOut) as e:
        wait_s = int(getattr(e, "retry_after", 30) or 30)
        job = {
            "created_at": iso_timestamp(),
            "attempts": 0,
            "next_try_at": iso_timestamp(),
            "text": text,
            "photo_path": raw_image_path,
            "reason": f"Network: {type(e).__name__}",
        }
        enqueue_alert(job)
    except TelegramError as e:
        wait_s = int(getattr(e, "retry_after", 30) or 30)
        job = {
            "created_at": iso_timestamp(),
            "attempts": 0,
            "next_try_at": iso_timestamp(),
            "text": text,
            "photo_path": raw_image_path,
            "reason": f"TelegramError: {type(e).__name__}",
        }
        enqueue_alert(job)
def build_bot(bot_token: str) -> Bot:
    return Bot(token=bot_token)
def send_text(bot: Bot, text: str, chat_id: str ) -> bool:
    try:
        bot.send_message(chat_id=chat_id, text=text)
        return True
    except TelegramError as e:
        print(f"[telegram] send_text failed: {e}")
        return False
def send_photo(bot: Bot, photo_path: Path, chat_id: str, caption: Optional[str] = None ) -> bool:
    try:
        with open(photo_path, "rb") as photo:
            bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)
            return True
    except FileNotFoundError:
        print(f"[telegram] image not found: {photo_path}")
        return False
    except TelegramError as e:
        print(f"[telegram] send_photo failed: {e}")
        return False

def format_event_message(event: dict[str,Any]) -> str:
    verdict = event.get("verdict", {} )

    return (
        f"SentientAI Alert\n"
        f"Time: {iso_timestamp()}\n"
        f"Risk Level: {verdict.get('level', 'UNKNOWN')}\n"
        f"Person Detected: {verdict.get('person_detected', False)}\n"
        f"Face Detected: {verdict.get('face_detected', False)}"
    )

def flush_outbox(max_send: int = 20) -> Dict[str, int]:
    cfg = load_telegram_config()
    bot = Bot(token=cfg.bot_token)

    if not OUTBOX_PATH.exists():
        return {"sent": 0, "kept": 0}
    lines = OUTBOX_PATH.read_text(encoding="utf-8").splitlines()
    jobs: List[Dict[str, Any]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            jobs.append(json.loads(ln))
        except Exception as e:
            continue
    sent = 0
    kept: List[Dict[str, Any]] = []
    for job in jobs:
        if sent >= max_send:
            kept.append(job)
            job["attempts"] = int(job.get("attempts", 0)) + 1
            continue
        try:
            send_now(bot, cfg.chat_id, job.get("text", ""), photo_path=job.get("photo_path"))
            sent += 1
        except (NetworkError, TimedOut, RetryAfter, TelegramError):
            kept.append(job)
            job["attempts"] = int(job.get("attempts", 0)) + 1
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(__import__("json").dumps(j, ensure_ascii=False) for j in kept) + ("\n" if kept else "")
    OUTBOX_PATH.write_text(payload, encoding="utf-8")

    return {"sent": sent, "kept": len(kept)}


