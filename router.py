import cv2
import time

from controller import Controller
from detectors import FaceDetector, PoseEstimator


def main():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_detector = FaceDetector()
    pose_detector = PoseEstimator()
    drone = Controller()

    rval, frame = vc.read() if vc.isOpened() else (False, None)
    while rval:
        rval, frame = vc.read()
        start = time.perf_counter()
        faces = face_detector.detect(frame)
        # poses = pose_detector.detect(frame)
        print(f"Detection took {time.perf_counter() - start} seconds")

        if faces:
            frame = face_detector.visualize(faces, frame)
            nX, nY = faces[0]["landmarks"][33]
            cv2.putText(frame,
                        f"Nose ({nX}, {nY})",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            (dX, dY) = drone.center_in_view(image=frame, x=nX, y=nY)
            cv2.putText(frame,
                        f"dX {dX[0]} {round(dX[1], 3)}",
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.putText(frame,
                        f"dY {dY[0]} {round(dY[1], 3)}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        cv2.imshow("preview", frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == '__main__':
    main()
