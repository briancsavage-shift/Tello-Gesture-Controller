import numpy as np
import dlib
import cv2
from imutils import face_utils
from typing import Dict, List, Any


class FaceDetector:
    def __init__(self):
        self.weights_filename = "shape_predictor_68_face_landmarks.dat"
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
        return image


class GestureRecognizer:
    def __init__(self):
        raise NotImplementedError
