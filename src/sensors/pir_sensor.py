from gpiozero import MotionSensor

from typing import Optional, Callable

import time


DEFAULT_PIR_PIN = 4


class PIRSensor:
    def __init__(self, pin=DEFAULT_PIR_PIN, warmup_seconds: float = 2.0):
        self.pin = pin
        self.warmup_seconds = warmup_seconds
        self.sensor = MotionSensor(self.pin)

    def warmup(self):
        time.sleep(self.warmup_seconds)

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        self.sensor.wait_for_motion(timeout=timeout)
        return bool(self.sensor.motion_detected)
    def wait_for_no_motion(self, timeout: Optional[float] = None) -> bool:
        self.sensor.wait_for_no_motion(timeout=timeout)
        return not bool(self.sensor.motion_detected)
    def is_motion_detected(self) -> bool:
        return bool(self.sensor.motion_detected)
    def is_no_motion_detected(self) -> bool:
        return not bool(self.sensor.motion_detected)

    def set_callbacks(
            self,
            on_motion: Optional[Callable[[], None]] = None,
            on_no_motion: Optional[Callable[[], None]] = None,
    ) -> None:
        self.sensor.when_motion = on_motion
        self.sensor.when_no_motion = on_no_motion

    def close(self) -> None:
        self.sensor.close()


