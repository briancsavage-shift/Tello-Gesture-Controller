import numpy as np
from djitellopy import Tello
from typing import Tuple


class Controller:
    """
        Wrapper class for DJITelloPy Module to fetch sensing data and
        execute instructions for the Tello drone.

    """
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

        self.inAir = False
        self.maxMagnitudeMove = 10
        self.maxMagnitudeRotation = 30

    def center_in_view(self,
                       image: np.ndarray,
                       x: int,
                       y: int) -> Tuple[Tuple[str, int], Tuple[str, int]]:
        H, W = image.shape[:2]
        mX = "counter_clockwise" if x < W // 2 else "clockwise"
        mY = "up" if y < H // 2 else "down"
        dX = (abs((W // 2) - x) / (W // 2)) * self.maxMagnitudeRotation
        dY = (abs((H // 2) - y) / (H // 2)) * self.maxMagnitudeMove
        return (mX, dX), (mY, dY)

    def takeoff(self):
        if not self.inAir:
            try:
                self.tello.takeoff()
                self.inAir = True
            except Exception as err:
                print(f"Error raised during takeoff: {err}")

    def land(self):
        if self.inAir:
            try:
                self.tello.land()
                self.inAir = False
            except Exception as err:
                print(f"Error raised during landing: {err}")
        else:
            print("Drone hasn't taken off. Can't execute land instruction.")

    def move(self, direction: str, magnitude: int):
        cases = {
            "forward": self.tello.move_forward,
            "back": self.tello.move_back,
            "left": self.tello.move_left,
            "right": self.tello.move_right,
            "up": self.tello.move_up,
            "down": self.tello.move_down,
            "clockwise": self.tello.rotate_clockwise,
            "counter_clockwise": self.tello.rotate_counter_clockwise,
            "none": lambda _: None
        }
        if self.inAir:
            if direction in cases:
                if 0 < magnitude < 100:
                    try:
                        return cases[direction](magnitude)
                    except Exception as err:
                        print(f"Error raised for instruction "
                              f"({direction}, {magnitude}): {err}")
                else:
                    print(f"Invalid magnitude: {magnitude}")
            else:
                print(f"Invalid Direction: {direction}")
        else:
            print("Drone hasn't taken off. Can't execute move instruction.")

    def view(self):
        return self.tello.get_frame_read().frame

    def metrics(self):
        return self.tello.get_current_state()
