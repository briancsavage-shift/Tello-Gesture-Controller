import numpy as np
from djitellopy import Tello
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Control:
    tello: Tello = Tello()
    inAir: bool = False
    maxMove: int = 30
    maxRotation: int = 30
    operations: Dict[str, Any] = field(default_factory=dict)

    def __init__(self):
        self.tello.connect()
        self.tello.streamon()
        self.operations = {
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

    def center_in_view(self,
                       image: np.ndarray,
                       x: int,
                       y: int) -> Tuple[Tuple[str, int], Tuple[str, int]]:
        """
            @Does

            @Params

            @Returns

        """
        H, W = image.shape[:2]
        mX = "counter_clockwise" if x < W // 2 else "clockwise"
        mY = "up" if y < H // 2 else "down"
        dX = (abs((W // 2) - x) / (W // 2)) * self.maxRotation
        dY = (abs((H // 2) - y) / (H // 2)) * self.maxMove
        return (mX, dX), (mY, dY)

    def takeoff(self):
        """
            @Does

            @Params

            @Returns

        """
        if self.inAir:
            print("Drone is already in air.")
            return self
        try:
            self.tello.land()
            self.inAir = False
            return self
        except Exception as err:  # TODO: Add specific exception handling
            print(f"Error raised during landing: {err}")

    def land(self):
        """
            @Does

            @Params

            @Returns

        """
        if not self.inAir:
            print("Drone hasn't taken off. Can't execute land instruction.")
        try:
            self.tello.land()
            self.inAir = False
            return self
        except Exception as err:
            print(f"Error raised during landing: {err}")

    def move(self, direction: str, magnitude: int):
        """
            @Does

            @Params

            @Returns

        """
        if not self.inAir:
            print("Drone hasn't taken off. Can't execute move instruction.")
            return self

        if direction not in self.operations:
            print(f"Invalid direction: {direction}")
            return self

        if 0 <= magnitude <= self.maxMove:
            try:
                self.operations[direction](magnitude)
                return self
            except Exception as err:
                print(f"Error raised for instruction "
                      f"({direction}, {magnitude}): {err}")
        else:
            print(f"Invalid magnitude: {magnitude}")

    def view(self):
        return self.tello.get_frame_read().frame

    def metrics(self):
        return self.tello.get_current_state()
