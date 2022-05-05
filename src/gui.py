import streamlit as st
import av
import time
import cv2
import os
import pandas as pd
from streamlit_webrtc import webrtc_streamer
from detectors import FaceDetector
from detectors import HandDetector
from controller import Controller


emojis = {
    "up": "☝️",
    "down": "👇",
    "left": "👈",
    "right": "👉",
    "clockwise": "⏩",
    "counterclockwise": "⏪",
    "backward": "🔴",
    "forward": "🟢",
}

class FrameProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.hand_detector = HandDetector()
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

            normalized_position = self.hand_detector.normalize(hands)

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

            if hands or faces:
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        return frame


def main():
    st.title("Tello Drone Navigation")

    with st.sidebar:
        st.subheader("User Inputs")
        mode = st.radio("Mode", ["Keyboard Controller",
                                 "Gesture Controller"]) #,
                                 #"Training Mode"])

        st.markdown("`On/Off`")
        l, r = st.columns(2)
        land = l.button("Land Drone")
        takeoff = r.button("Takeoff Drone")
        st.markdown("-----------")

        cmds = {}
        if mode in ["Keyboard Controller", "Training Mode"]:
            st.markdown("`Keyboard Controller`")
            rows = [st.columns(4), st.columns(4)]
            for i, command in enumerate(emojis.keys()):
                cmds.update({
                    command: rows[i % 2][i // 2].button(emojis[command])
                })
        else:
            st.subheader("Gesture Controller from Webcam")

    webrtc_streamer(key="sample", video_processor_factory=FrameProcessor)

    # if mode == "Training Mode":
    #     st.markdown("---------")
    #     st.subheader("Training Data Collection")
    #
    #     l, r = st.columns(2)
    #     l.button("Delete Last Frame")
    #     r.button("Save Training Data")
    #
    #     # for i, command in enumerate(cmds.keys()):
    #     #     if cmds[command]:
    #     #         # if st.session_state["last_loc"] != "None":
    #         #         st.session_state["label_record"].append(
    #     #             {
    #     #                 "command": list(cmds.keys())[i],
    #     #                 "coordinates": None,
    #     #             })
    #                 # st.session_state["last_loc"] = "None"
    #
    #     st.write("Data: ")
    #     st.write(st.session_state["label_record"])


# def save_image(command: str):
#     #if st.session_state["last_loc"] != "None":
#     st.session_state["label_record"].append(
#         {
#             "command": command,
#             "coordinates": holder,
#         })





if __name__ == "__main__":
    main()