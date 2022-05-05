import numpy as np
import mediapipe as mp
import dlib
import cv2
from imutils import face_utils
from typing import Dict, List, Tuple, Any, NamedTuple, Optional


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



class HandDetector:
    def __init__(self,
                 min_detection_confidence: Optional[float] = 0.5,
                 min_tracking_confidence: Optional[float] = 0.5):
        self.min_detect: float = min_detection_confidence
        self.min_track: float = min_tracking_confidence
        self.model_complexity: int = 1

        self.mp_drawings = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def detect(self, image: np.ndarray) -> NamedTuple:
        """
        Detect hands within an image.
        :param image:
        :return:
        """
        with self.mp_hands.Hands(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detect,
                min_tracking_confidence=self.min_track) as hands:
            return hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def visualize(self, results: NamedTuple, image: np.ndarray) -> np.ndarray:
        """
        Visualize the results of the hand detection pipeline.
        :param results:
        :param image:
        :return:
        """
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawings.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)