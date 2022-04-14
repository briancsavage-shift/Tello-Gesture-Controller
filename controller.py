from djitellopy import Tello


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
            "forward": self.tello.forward,
            "backward": self.tello.backward,
            "left": self.tello.left,
            "right": self.tello.right,

            "clockwise": self.tello.clockwise,
            "counter_clockwise": self.tello.counter_clockwise,
            "up": self.tello.up,
            "down": self.tello.down,
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
