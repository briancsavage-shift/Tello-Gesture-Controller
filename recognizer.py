import cv2
import os
# import time
import uuid
from typing import Dict, List, Any, Optional
# import PyAudio
# import speech_recognition as sr


class SpeechRecognizer:
    def __init__(self):
        pass


class SignRecognizer:
    def __init__(self):
        self.images: os.path = os.path.join(os.getcwd(), 'images')
        self.labels: Dict[str, Dict[str, Any]] = {
            "up": {
                "uses_both_hands": False,
                "description": "Pointer UP",
            },
            "down": {
                "uses_both_hands": False,
                "description": "Pointer DOWN",
            },
            "left": {
                "uses_both_hands": False,
                "description": "Pointer LEFT",
            },
            "right": {
                "uses_both_hands": False,
                "description": "Pointer RIGHT",
            },
            "forward": {
                "uses_both_hands": True,
                "description": "Parallel UP",
            },
            "backward": {
                "uses_both_hands": True,
                "description": "Parallel DOWN",
            },
            "clockwise": {
                "uses_both_hands": False,
                "description": "Pointer and Pinky UP, Thumb RIGHT",
            },
            "counterclockwise": {
                "uses_both_hands": False,
                "description": "Pointer and Pinky UP, Thumb LEFT",
            },
            "no_sign": {
                "uses_both_hands": False,
                "description": "NONE",
            }
        }

    def record_training_data(self, count_per_class: int = 5) -> None:
        # cv2.namedWindow("preview")
        # vc: cv2.VideoCapture = cv2.VideoCapture(0)

        for label, info in self.labels.items():
            print(f"[ Collecting {count_per_class} images for {label} ]")
            print(f"**label:{label}** {info['description']} -> uses " +
                  ("both hands" if info['uses_both_hands'] else "either hand"))

            # label_directory = os.path.join(self.images, label)
            # if os.path.isdir(label_directory):
            #     for file in os.listdir(label_directory):
            #         os.remove(os.path.join(label_directory, file))
            # else:
            #     os.mkdir(label_directory)
            #
            # for i in range(count_per_class):
            #     image_path = os.path.join(label_directory,
            #                               str(uuid.uuid4()) + '.jpg')


if __name__ == "__main__":
    recognizer = SignRecognizer()
    recognizer.record_training_data()
