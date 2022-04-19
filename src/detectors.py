import numpy as np
import dlib
import cv2
import mediapipe as mp

from imutils import face_utils
from typing import Dict, List, Tuple, Any


class FaceDetector:
    def __init__(self):
        self.weights_filename = "../models/face-detection" \
                                "/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.weights_filename)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        features = []
        faces = self.detector(image, 0)
        for face in faces:
            landmarks = self.predictor(image, face)
            features.append({
                "landmarks": face_utils.shape_to_np(landmarks),
                "faceAreas": face.area(),
                "faceEdges": ((face.left(), face.top()),
                              (face.right(), face.bottom()))
            })
        return features

    @staticmethod
    def visualize(features: List[Dict[str, Any]],
                  image: np.ndarray) -> np.ndarray:
        for face in features:
            for x, y in face["landmarks"]:
                image = cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            cX, cY = face["landmarks"][33]
            mX, mY = image.shape[1] // 2, image.shape[0] // 2

            image = cv2.line(image, (cX, cY), (mX, mY), (255, 0, 0), 2)
            image = cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            image = cv2.circle(image, (mX, mY), 4, (0, 0, 255), -1)
        return image

    @staticmethod
    def midpoint(face_edges: Tuple[Tuple[int, int],
                                   Tuple[int, int]]) -> Tuple[float, float]:
        (left, top), (right, bottom) = face_edges
        return (left + right) / 2, (top + bottom) / 2


class PoseEstimator:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_detector = mp.solutions.holistic

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        features = []
        with self.mp_detector.Holistic(min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5) as detector:

            RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = detector.process_image(RGB)
            print(res)

        return features


class GestureRecognizer:
    def __init__(self):
        raise NotImplementedError
