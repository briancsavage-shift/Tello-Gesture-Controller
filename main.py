import time
import pprint
import numpy as np
from typing import Tuple
from controller import Controller
from detectors import FaceDetector


def main():
    tello = Controller()
    detector = FaceDetector()
    status = tello.metrics()
    log(status)

    # tello.takeoff()
    # tello.move(direction="up", magnitude=70)

    tracking = None
    image = np.ndarray(shape=(480, 640, 3), dtype=np.uint8)

    try:
        while True:
            if tracking is None:
                # tello.move(direction="clockwise", magnitude=30)
                pass
            else:
                X, Y = tracking
                (direction, magnitude) = tello.center_in_view(image=image,
                                                              x=X,
                                                              y=Y)
                # tello.move(direction=direction, magnitude=magnitude)

            image = tello.view()
            faces = detector.detect(image)
            tracking = midpoint(faces[0]["faceEdges"]) if faces else None
            print(f"Tracking: {tracking}" if tracking else "No face detected")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        # tello.land()

    return


def log(msg: str):
    pprint.pprint(msg, indent=4)


def midpoint(face_edges: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple:
    (left, top), (right, bottom) = face_edges
    return (left + right) / 2, (top + bottom) / 2


if __name__ == '__main__':
    main()
