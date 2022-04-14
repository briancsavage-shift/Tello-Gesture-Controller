from typing import Tuple
from controller import Controller
from detectors import FaceDetector
import pprint


def main():
    tello = Controller()
    detector = FaceDetector()
    status = tello.metrics()
    log(status)

    tello.takeoff()
    tello.move(direction="up", magnitude=50)
    tracking = None

    while tracking is None:
        if tracking is None:
            tello.move(direction="clockwise", magnitude=30)
        else:
            (direction, magnitude) = tello.center_in_view(tracking)
            tello.move(direction=direction, magnitude=magnitude)

        image = tello.view()
        faces = detector.detect(image)
        tracking = midpoint(faces[0]["faceEdges"]) if faces else None
        print(f"Tracking: {tracking}" if tracking else "No face detected")

    tello.land()
    return


def log(msg: str):
    pprint.pprint(msg, indent=4)


def midpoint(face_edges: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple:
    (left, top), (right, bottom) = face_edges
    return (left + right) / 2, (top + bottom) / 2


if __name__ == '__main__':
    main()
