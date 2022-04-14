from controller import Controller
from detectors import FaceDetector
import pprint
import cv2


def main():
    log = lambda x: pprint.pprint(x, indent=4)

    tello = Controller()
    detector = FaceDetector()

    status = tello.metrics()
    log(status)

    tello.takeoff()

    tello.move(direction="up", magnitude=20)
    tello.move(direction="clockwise", magnitude=20)

    # count = 15
    #
    # while count > 0:
    #     img = tello.view()
    #     features = detector.detect(img)
    #     annotated = detector.visualize(features, img)
    #
    #     log(features)
    #
    #     cv2.imshow("Drone View", annotated)
    #     cv2.waitKey(1)
    #     count -= 1

    tello.land()
    return


if __name__ == '__main__':
    main()
