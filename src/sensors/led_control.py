from gpiozero import LED
import time
import board
import neopixel
from typing import Optional


DEFAULT_LED_PIN = 17


pixels1 = neopixel.NeoPixel(board.D17, 55, brightness=1)





class LEDControl:
    def __init__(self, pin: int = DEFAULT_LED_PIN):
        self.led = LED(pin)
        self.pixels = pixels1
    def on(self) -> None:
        self.led.on()

        x = 0
        self.pixels.fill((255, 255, 255))
        self.pixels[10] = (255, 255, 255)
        time.sleep(4)
        while x < 35:
            self.pixels[x] = (255, 255, 255)
            self.pixels[x - 5] = (255, 255, 255)
            self.pixels[x - 10] = (255, 255, 255)

            x += 1
            time.sleep(0.05)
        while x > -15:
            self.pixels[x] = (255, 255, 255)
            self.pixels[x + 5] = (255, 255, 255)
            self.pixels[x + 10] = (255, 255, 255)
            x -= 1
            time.sleep(0.05)
    def off(self) -> None:
        self.led.off()
        time.sleep(4)

        self.pixels.fill((0, 0, 0))
    def blink(self) -> None:
        self.led.blink(on_time=0.2, off_time=0.2, background=True)
    def close(self) -> None:
        self.led.close()
        self.pixels.fill((0, 0, 0))