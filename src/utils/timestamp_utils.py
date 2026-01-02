from datetime import datetime
from zoneinfo import ZoneInfo



def get_timezone(tz_name: str = "America/Detroit" ) -> ZoneInfo:
    return ZoneInfo(tz_name)

def now(tz_name: str = "America/Detroit") -> datetime:
    dt = ZoneInfo(tz_name)
    return datetime.now(dt)

def timestamp_for_filename(dt: datetime | None = None, tz_name: str = "America/Detroit") -> str:
    dt = dt or now(tz_name=tz_name)
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

def date_folder(dt: datetime | None = None, tz_name: str = "America/Detroit") -> str:
    dt = dt or now(tz_name=tz_name)
    return dt.strftime("%Y-%m-%d")

def iso_timestamp(dt: datetime | None = None, tz_name: str = "America/Detroit") -> str:
    dt = dt or now(tz_name=tz_name)
    return dt.isoformat(timespec="seconds")

def parse_iso_timestamp(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def unique_event_id(prefix="evt", tz_name: str = "America/Detroit") -> str:
    return f"{prefix}_{timestamp_for_filename(tz_name=tz_name)}"

if __name__ == "__main__":
    print(iso_timestamp())
    print(timestamp_for_filename())
    print(date_folder())
    print(unique_event_id())

