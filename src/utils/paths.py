from pathlib import Path
import datetime
import os



def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



BASE_DIR = Path(__file__).resolve().parents[2]

SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "images" / "raw"
PROCESSED_DIR = DATA_DIR / "images" / "processed"

LOG_DIR = BASE_DIR / "logs"



def ensure_directories():

    for d in [CONFIG_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_raw_image_path(timestamp):
    return f"{RAW_DIR}/{timestamp}.jpg"



def get_processed_image_path(timestamp):
    return f"{PROCESSED_DIR}/{timestamp}.jpg"


def get_result_json_path(timestamp):
    return f"{LOG_DIR}/{timestamp}.json"
ensure_directories()
