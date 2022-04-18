import cv2
import os
import time
import uuid
from typing import Dict, List, Any, Optional
# import PyAudio
# import speech_recognition as sr


class SpeechRecognizer:
    def __init__(self):
        pass


class SignRecognizer:
    def __init__(self):
        self.data_dir: os.path = os.path.join(os.getcwd(), "..", "data")
        self.annotations: os.path = os.path.join(self.data_dir, 'annotations')
        self.images: os.path = os.path.join(self.data_dir, 'images')

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

    def record_training_data(self, count_per_class: Optional[int] = 25) -> None:
        cv2.namedWindow("preview")
        cap: cv2.VideoCapture = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imshow("preview", frame)

        for label, info in self.labels.items():
            print(f"[ Collecting {count_per_class} images for {label} ]")
            print(f"**label:{label}** {info['description']} -> uses " +
                  ("both hands" if info['uses_both_hands'] else "either hand"))
            time.sleep(5)

            label_directory = os.path.join(self.images, label)
            if os.path.isdir(label_directory):
                for file in os.listdir(label_directory):
                    os.remove(os.path.join(label_directory, file))
            else:
                os.mkdir(label_directory)

            for i in range(count_per_class):
                print(f"\t[ Collecting image {i + 1} of {count_per_class} ]")
                image_path = os.path.join(label_directory,
                                          label + '-' +
                                          str(uuid.uuid4()) + '.jpg')
                ret, frame = cap.read()
                cv2.imwrite(image_path, frame)
                cv2.imshow("preview", frame)
                time.sleep(2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()

    def make_label_map(self) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for i, (label, info) in enumerate(self.labels.items()):
            labels.append({"name": label, "id": i + 1})
        return labels

    def make_tf_records(self) -> bool:
        #!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l
        # {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}

        #!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l
        # {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}

        scripts = lambda split: ["python",
                                 "utils/generate_tfrecord.py", "-x",
                                 self.images + f"/{split}", "-l",
                                 self.annotations + "/label_map.pbtxt", "-o",
                                 self.annotations + f"/{split}.record"]

        for directory in ["train", "test"]:
            print(f"Generating {directory} records")
        return True




    def train(self) -> None:
        # TODO: Variables

        # TODO: Create label map
        labels = self.make_label_map()

        # TODO: Create TF Records

        # TODO: Download Pretrained TF Models from Tensorflow Model Zoo
        pass


if __name__ == "__main__":
    recognizer = SignRecognizer()
    recognizer.record_training_data()
