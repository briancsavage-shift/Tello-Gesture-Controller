import pprint
import numpy as np
import cv2
from src.controller import Controller
from src.detectors import FaceDetector


def main():
    cv2.namedWindow("preview")
    tello = Controller()
    detector = FaceDetector()
    status = tello.metrics()
    log(status)

    tello.takeoff()
    tello.move(direction="up", magnitude=50)

    tracking = None
    image = np.ndarray(shape=(720, 960, 3), dtype=np.uint8)

    while True:
        if tracking is None:
            tello.move(direction="clockwise", magnitude=15)
        else:
            X, Y = tracking
            (dX, dY) = tello.center_in_view(image=image, x=int(X), y=int(Y))
            tello.move(direction=dX[0], magnitude=dX[1])

        image = tello.view()
        faces = detector.detect(cv2.resize(image, (image.shape[0] // 3,
                                                   image.shape[1] // 3)))
        tracking = detector.midpoint(faces[0]["faceEdges"]) if faces else None
        annotated = detector.visualize(faces, image)
        print(f"Tracking: {tracking}" if tracking else "No face detected")

        cv2.imshow("preview", annotated)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
        # time.sleep(1)
        # tello.land()

    return


def log(msg: str):
    pprint.pprint(msg, indent=4)





if __name__ == '__main__':
    main()
