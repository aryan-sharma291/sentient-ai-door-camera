import json
from pathlib import Path
import tempfile
import os
import datetime
import numpy as np


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    text = p.read_text(encoding="utf-8")
    return json.loads(text)


def write_json(path, data, pretty=True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"ensure_ascii": False, "default": to_jsonable}
    if pretty:
        kwargs["indent"] = 2

    payload = json.dumps(data, **kwargs)
    p.write_text(payload, encoding="utf-8")


def append_jsonl(path, record):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, default=to_jsonable, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe_write_json(path, data, pretty=True):
    """
    Atomic-ish write:
    - Write to a temp file in the SAME directory
    - fsync
    - os.replace to swap it in
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        data,
        default=to_jsonable,
        ensure_ascii=False,
        indent=2 if pretty else None,
    )

    # temp file must be in same directory for safest replace
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            tf.write(payload)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_name, p)  # atomic replace on most OS/filesystems
    finally:
        # If something failed before replace, clean temp
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)

    # numpy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # datetime types
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # last resort: let json raise TypeError for unknown objects
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



