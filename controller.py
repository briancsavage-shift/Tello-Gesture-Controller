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

    @staticmethod
    def center_in_view(image: np.ndarray,
                       x: int,
                       y: int) -> Tuple[str, int]:
        move = "None"
        H, W = image.shape[:2]
        if x < W // 2:
            move = "counter_clockwise"
        elif x > W // 2:
            move = "clockwise"

        X, Y = (W // 2, H // 2)
        magnitude = np.sqrt((x - X) ** 2 + (y - Y) ** 2)
        max_value = np.sqrt((W // 2) ** 2 + (H // 2) ** 2)
        normalized = np.multiply(np.divide(magnitude, max_value), 100)

        print(f"Computed Magnitude: {normalized}")
        return move, 0

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
