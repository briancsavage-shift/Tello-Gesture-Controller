import cv2
import numpy as np
from detectors import HandDetector
from gesture_recognizer import GestureRecognizer


def main():
    emojis = [
        "☝️",
        "👇",
        "👈",
        "👉",
        "🔴",
        "🟢",
        "⏩",
        "⏪",
    ]

    hand_detector = HandDetector()
    gesture_recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hands = hand_detector.detect(frame)
        image = hand_detector.visualize(hands, frame)

        hand = hands.multi_hand_landmarks[0] if \
               hands.multi_hand_landmarks else None

        if hand is not None:
            probas = gesture_recognizer.predict(hand)
            print(probas)

        cv2.imshow('frame', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





    return

if __name__ == "__main__":
    main()
