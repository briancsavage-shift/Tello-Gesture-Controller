import cv2
import os
import time
import uuid


class SignRecognizer:
    def __init__(self):
        self.images_path = os.path.join(os.getcwd(), 'images')

        self.labels = ["up",
                       "down",
                       "left",
                       "right",
                       "forward",
                       "backward",
                       "clockwise",
                       "counterclockwise",
                       "no_sign"]

    def collect_images(self, count_per_class: int = 5):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        for label in self.labels:
            label_directory = os.path.join(self.images_path, label)
            if os.path.isdir(label_directory):
                for file in os.listdir(label_directory):
                    os.remove(os.path.join(label_directory, file))
            else:
                os.mkdir(label_directory)

            for i in range(count_per_class):
                image_path = os.path.join(label_directory,
                                          str(uuid.uuid4()) + '.jpg')

