import streamlit as st
import av
import time
import cv2
import os
import pandas as pd
import numpy as np
from streamlit_webrtc import webrtc_streamer
from gesture_recognizer import GestureRecognizer
from detectors import FaceDetector
from detectors import HandDetector
from controller import Controller
from recognizer import SignRecognizer

emojis = {
    "up": "☝️",
    "down": "👇",
    "left": "👈",
    "right": "👉",
    "backward": "🔴",
    "forward": "🟢",
    "clockwise": "⏩",
    "counterclockwise": "⏪",
}

if "gesture" not in st.session_state:
    st.session_state["gesture"] = None




class FrameProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.hand_detector = HandDetector()
        self.gesture_recognizer = GestureRecognizer()
        self.sign_recognizer = SignRecognizer()
        self.drone = Controller()

    def recv(self, frame):
        if frame:
            img = av.VideoFrame.to_ndarray(frame, format="bgr24")

            t = time.perf_counter()
            faces = self.face_detector.detect(img)
            print(f"Face Detection took"
                  f" {round(time.perf_counter() - t, 4)} seconds")

            t = time.perf_counter()
            hands = self.hand_detector.detect(img)
            print(f"Hands Detection took"
                  f" {round(time.perf_counter() - t, 4)} seconds")

            if faces:
                annotated = self.face_detector.visualize(faces, img)
                nX, nY = faces[0]["landmarks"][33]
                cv2.putText(annotated,
                            f"Nose ({nX}, {nY})",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

                (dX, dY) = self.drone.center_in_view(image=img, x=nX, y=nY)
                cv2.putText(annotated,
                            f"dX {dX[0]} {round(dX[1], 3)}",
                            (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                cv2.putText(annotated,
                            f"dY {dY[0]} {round(dY[1], 3)}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            else:
                annotated = img

            if hands:
                annotated = self.hand_detector.visualize(hands, annotated)
                hand = hands.multi_hand_landmarks[0] if \
                       hands.multi_hand_landmarks else []

                # if hand:
                #     signs, visualized = self.sign_recognizer.inference(img)
                #     print(signs)

                    # probas = self.gesture_recognizer.predict(hand)
                    # print(list(emojis.values())[np.argmax(probas)])
                    # st.session_state.gesture = list(emojis.values())[
                    #                             np.argmax(probas)
                    # ]



            if hands or faces:
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        return frame


def main():
    st.title("Tello Drone Navigation")
    drone = Controller()

    with st.sidebar:
        st.subheader("User Inputs")
        mode = st.radio("Mode", ["Keyboard Controller", "Gesture Controller"])

        st.markdown("`On/Off`")
        l, r = st.columns(2)
        land = l.button("Land Drone", key="land")
        if land:
            print("Land called!")
            drone.land()

        takeoff = r.button("Takeoff Drone", key="takeoff")
        if takeoff:
            print("Takeoff called!")
            drone.takeoff()
        st.markdown("-----------")

        cmds = {}
        if mode in ["Keyboard Controller", "Training Mode"]:
            st.markdown("`Keyboard Controller`")
            rows = [st.columns(4), st.columns(4)]
            for i, command in enumerate(emojis.keys()):
                cmds.update({
                    command: rows[i % 2][i // 2].button(emojis[command])
                })

            for key in emojis.keys():
                if cmds[key]:
                    drone.move(key, magnitude=20)


        else:
            st.subheader("Gesture Controller from Webcam")


    #
    # cv2.namedWindow("preview")
    # vc = cv2.VideoCapture(0)
    #
    # face_detector = FaceDetector()
    # sign_recognizer = SignRecognizer()
    # drone = Controller()
    #
    # rval, frame = vc.read() if vc.isOpened() else (False, None)
    # while rval:
    #     rval, frame = vc.read()
    #     start = time.perf_counter()
    #     faces = face_detector.detect(frame)
    #     # signs, visualized = sign_recognizer.inference(frame)
    #     # print(signs)
    #
    #
    #
    #
    #
    #     print(f"Detection took {round(time.perf_counter() - start, 4)} seconds")
    #
    #     if faces:
    #         frame = face_detector.visualize(faces, frame)
    #         nX, nY = faces[0]["landmarks"][33]
    #         cv2.putText(frame,
    #                     f"Nose ({nX}, {nY})",
    #                     (10, 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.6, (0, 0, 255), 2)
    #
    #         (dX, dY) = drone.center_in_view(image=frame, x=nX, y=nY)
    #         cv2.putText(frame,
    #                     f"dX {dX[0]} {round(dX[1], 3)}",
    #                     (10, 45),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.6, (0, 0, 255), 2)
    #         cv2.putText(frame,
    #                     f"dY {dY[0]} {round(dY[1], 3)}",
    #                     (10, 70),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.6, (0, 0, 255), 2)
    #
    #     cv2.imshow("preview", frame)
    #     key = cv2.waitKey(20)
    #     if key == 27:  # exit on ESC
    #         break
    #
    # cv2.destroyWindow("preview")
    # vc.release()

    # webrtc_streamer(key="sample", video_processor_factory=FrameProcessor)

    # if st.session_state.gesture is not None:
    #     st.code(f"Gesture={st.session_state.gesture}")

    # if mode == "Keyboard Controller":
    #     with st.spinner("Connecting to Tello Drone..."):
    #         drone = Controller()
    #

if __name__ == "__main__":
    main()